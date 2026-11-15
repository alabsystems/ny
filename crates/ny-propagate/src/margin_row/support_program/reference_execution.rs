// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Original reference arithmetic, independently of root/tableau precision.
//! Records are retained inside the paid construction owner, never accepted
//! from an external receipt. Ceased endpoint buffers retain their usual liveness.

use std::mem::size_of;
use std::time::Instant;

use ny_core::Result;

use super::ir::{checkpoint, invalid, refusal};
use super::precision::ReferenceF32Receipt;
use super::{CertifiedTensorBox, OriginalRoot};
use crate::certify::work::{LeaseClass, WorkRequest};
use crate::margin_row::net::TwinOp;

/// Reference endpoint arithmetic only. Neither choice delivers root/tableau
/// binary32 kernels or binary32 endpoint storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreparedReferenceArithmetic {
    #[default]
    F64,
    /// Binary32 nominal Conv/ChannelAffine dots. Other original operators
    /// and outward certificate/endpoints remain binary64.
    Binary32Certified,
}

/// Completed reference observations, not an authority to import numerical
/// arrays, publish a verdict, or claim tableau work or measured performance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreparedReferenceReceipt {
    pub arithmetic: PreparedReferenceArithmetic,
    pub expected_conv_ops: usize,
    pub completed_conv_ops: usize,
    pub expected_affine_ops: usize,
    pub completed_affine_ops: usize,
    pub binary32_conv_dot_ops: usize,
    pub binary32_affine_dot_ops: usize,
    pub multiply_add_pairs: usize,
    pub converted_cells: usize,
    pub subnormal_casts_normalized: usize,
    pub constant_outputs: usize,
}

struct Step {
    op: usize,
    input: usize,
    tensor: usize,
    conv: bool,
    binary32: Option<ReferenceF32Receipt>,
}

pub(super) struct ReferenceExecution {
    root: OriginalRoot,
    arithmetic: PreparedReferenceArithmetic,
    steps: Vec<Step>,
    capacity: usize,
    deadline: Instant,
    receipt: PreparedReferenceReceipt,
}

impl ReferenceExecution {
    pub(super) fn allocation_bytes(capacity: usize) -> Result<usize> {
        capacity.checked_mul(size_of::<Step>()).ok_or_else(overflow)
    }

    /// Called only after construction admits the record capacity as part of
    /// its persistent allocation; this owner is never separately readmitted.
    pub(super) fn new(
        root: OriginalRoot,
        arithmetic: PreparedReferenceArithmetic,
        capacity: usize,
        deadline: Instant,
    ) -> Result<Self> {
        checkpoint(root.budget(), Some(deadline))?;
        let mut steps = Vec::new();
        steps
            .try_reserve_exact(capacity)
            .map_err(|_| invalid("reference record allocation"))?;
        if steps.capacity() > capacity {
            return Err(invalid(
                "reference actual record capacity exceeds admission",
            ));
        }
        Ok(Self {
            root,
            arithmetic,
            steps,
            capacity,
            deadline,
            receipt: PreparedReferenceReceipt {
                arithmetic,
                ..Default::default()
            },
        })
    }

    /// Only genuine construction calls this immediately after its selected
    /// original producer completes and transfers the paid endpoint owner.
    pub(super) fn record(
        &mut self,
        op: usize,
        first: &CertifiedTensorBox,
        output: &CertifiedTensorBox,
        binary32: Option<ReferenceF32Receipt>,
    ) -> Result<()> {
        let (input, conv) = original(&self.root, op)?;
        if !self.root.same_root(first.root())
            || !self.root.same_root(output.root())
            || first.tensor() != input
            || output.tensor() != op.checked_add(1).ok_or_else(overflow)?
            || binary32.is_some()
                != (self.arithmetic == PreparedReferenceArithmetic::Binary32Certified)
            || self.steps.last().is_some_and(|last| last.op >= op)
            || self.steps.len() >= self.capacity
        {
            return Err(invalid("reference execution original transfer changed"));
        }
        self.deadline = self
            .root
            .deadline(Some(self.deadline))?
            .unwrap_or(self.deadline);
        if let Some(at) = first.deadline()? {
            self.deadline = self.deadline.min(at);
        }
        if let Some(at) = output.deadline()? {
            self.deadline = self.deadline.min(at);
        }
        checkpoint(self.root.budget(), Some(self.deadline))?;
        self.steps.push(Step {
            op,
            input,
            tensor: output.tensor(),
            conv,
            binary32,
        });
        Ok(())
    }

    pub(super) fn finish(&mut self, root: &OriginalRoot, at: Instant) -> Result<()> {
        self.deadline = self.deadline.min(at);
        self.receipt = self.scan(root, at)?;
        Ok(())
    }

    pub(super) fn receipt(&self) -> PreparedReferenceReceipt {
        self.receipt
    }

    /// Constant-time retained-owner checks never acquire a fresh lease. Full
    /// membership validation belongs to paid authority-consuming boundaries.
    pub(super) fn checkpoint(&self, root: &OriginalRoot, at: Instant) -> Result<()> {
        self.check_owner(root, at)?;
        let r = self.receipt;
        if r.arithmetic != self.arithmetic
            || r.expected_conv_ops != r.completed_conv_ops
            || r.expected_affine_ops != r.completed_affine_ops
            || add(r.completed_conv_ops, r.completed_affine_ops)? != self.steps.len()
            || r.binary32_conv_dot_ops > r.completed_conv_ops
            || r.binary32_affine_dot_ops > r.completed_affine_ops
            || (self.arithmetic == PreparedReferenceArithmetic::F64
                && (r.binary32_conv_dot_ops != 0
                    || r.binary32_affine_dot_ops != 0
                    || r.multiply_add_pairs != 0
                    || r.converted_cells != 0
                    || r.subnormal_casts_normalized != 0
                    || r.constant_outputs != 0))
        {
            return Err(invalid("reference execution arithmetic/header changed"));
        }
        Ok(())
    }

    pub(super) fn record_bytes(&self) -> Result<usize> {
        Self::allocation_bytes(self.capacity)
    }

    pub(super) fn validate(&self, root: &OriginalRoot, at: Instant) -> Result<()> {
        self.checkpoint(root, at)?;
        if self.scan(root, at)? != self.receipt {
            return Err(invalid("reference execution receipt changed"));
        }
        Ok(())
    }

    fn check_owner(&self, root: &OriginalRoot, requested: Instant) -> Result<()> {
        let at = requested.min(self.deadline);
        if !self.root.same_root(root)
            || self.steps.capacity() > self.capacity
            || self.steps.len() > self.steps.capacity()
        {
            return Err(invalid("reference execution foreign source/capacity"));
        }
        checkpoint(root.budget(), Some(at))?;
        if self.arithmetic == PreparedReferenceArithmetic::Binary32Certified {
            super::precision::qualify()?;
        } else {
            ny_core::require_f64_interval_proof_environment()?;
        }
        checkpoint(root.budget(), Some(at))
    }

    fn scan(&self, root: &OriginalRoot, requested: Instant) -> Result<PreparedReferenceReceipt> {
        let at = requested.min(self.deadline);
        self.check_owner(root, at)?;
        let net = root.model().net();
        let work = u64::try_from(net.i_gemm1)
            .ok()
            .and_then(|n| n.checked_mul(128))
            .ok_or_else(overflow)?;
        let mut lease = root
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    work,
                    ..Default::default()
                },
            )
            .map_err(refusal)?;
        let at = lease.deadline().map_or(at, |d| at.min(d));
        let mut result = PreparedReferenceReceipt {
            arithmetic: self.arithmetic,
            ..Default::default()
        };
        let mut cursor = 0usize;
        for (op, kind) in net.ops.iter().take(net.i_gemm1).enumerate() {
            if op % 256 == 0 {
                checkpoint(root.budget(), Some(at))?;
            }
            if !matches!(kind, TwinOp::Conv(_) | TwinOp::ChannelAffine { .. }) {
                continue;
            }
            let (input, conv) = original(root, op)?;
            let step = self
                .steps
                .get(cursor)
                .ok_or_else(|| invalid("missing original reference execution"))?;
            if step.op != op
                || step.input != input
                || step.tensor != op + 1
                || step.conv != conv
                || step.binary32.is_some()
                    != (self.arithmetic == PreparedReferenceArithmetic::Binary32Certified)
            {
                return Err(invalid("reference execution incomplete original coverage"));
            }
            if conv {
                result.expected_conv_ops = add(result.expected_conv_ops, 1)?;
                result.completed_conv_ops = add(result.completed_conv_ops, 1)?;
            } else {
                result.expected_affine_ops = add(result.expected_affine_ops, 1)?;
                result.completed_affine_ops = add(result.completed_affine_ops, 1)?;
            }
            if let Some(r) = step.binary32 {
                result.binary32_conv_dot_ops = add(result.binary32_conv_dot_ops, r.conv_ops)?;
                result.binary32_affine_dot_ops = add(result.binary32_affine_dot_ops, r.affine_ops)?;
                result.multiply_add_pairs = add(result.multiply_add_pairs, r.multiply_add_pairs)?;
                result.converted_cells = add(result.converted_cells, r.converted_cells)?;
                result.subnormal_casts_normalized = add(
                    result.subnormal_casts_normalized,
                    r.subnormal_casts_normalized,
                )?;
                result.constant_outputs = add(result.constant_outputs, r.constant_outputs)?;
            }
            cursor = add(cursor, 1)?;
        }
        if cursor != self.steps.len() {
            return Err(invalid("extra original reference execution"));
        }
        checkpoint(root.budget(), Some(at))?;
        lease.complete().map_err(refusal)?;
        Ok(result)
    }
}

fn original(root: &OriginalRoot, op: usize) -> Result<(usize, bool)> {
    if op >= root.model().net().i_gemm1 {
        return Err(invalid("reference execution outside original trunk"));
    }
    match root.model().net().ops.get(op) {
        Some(TwinOp::Conv(c)) => Ok((c.input, true)),
        Some(TwinOp::ChannelAffine { input, .. }) => Ok((*input, false)),
        _ => Err(invalid(
            "reference execution requires original Conv/ChannelAffine",
        )),
    }
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(overflow)
}
fn overflow() -> ny_core::NyError {
    invalid("reference execution arithmetic overflow")
}

#[cfg(test)]
#[path = "reference_execution_tests.rs"]
mod tests;
