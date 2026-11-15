// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit, paid affine snapshots for the first original head Gemm.
//!
//! These are preactivation rows in original input coordinates, not final
//! output-class functionals. The existing ordinary producer supplies both
//! directions, coefficient errors and independent affine constants. This
//! reader does not alter the consuming RootEval route or copy its row planes.

use std::time::Instant;

use ny_core::Result;

use super::cli_proof::PreparedRootSession;
use super::ir::{invalid, refusal};
use super::ordinary_pass::{
    build_prepared_ordinary_rows, PreparedOrdinaryRowView, PreparedOrdinaryRows,
};
use super::root_adapter::project_prepared_root;
use crate::certify::work::{LeaseClass, RetainedOwner, WorkRequest};
use crate::margin_row::BackwardIntermRuntimePolicy;

const OWNER_WORK: u64 = 512;

struct HeadPreactivationRowsInner {
    rows: PreparedOrdinaryRows,
    deadline: Instant,
}

fn owner_bytes() -> Result<usize> {
    RetainedOwner::<HeadPreactivationRowsInner>::metadata_bytes().map_err(refusal)
}

/// Shared, immutable first-head preactivation affine enclosures.
///
/// This standalone opt-in product builds its own fully paid projection and
/// complete ordinary row owner. It neither shares nor changes a separately
/// requested consuming RootEval transport. Clones share these new reservations
/// and keep the exact original declared source/domain alive until last drop.
/// Equal model parameters, digests or limits cannot replace that source.
///
/// Numeric snapshots are conditional on the original declared model family
/// and input box. They confer no verdict, proof grade, phase feasibility,
/// source-file authentication or publication eligibility. Consumers pay for
/// their own scans, copies, downstream outward arithmetic and solver work.
#[derive(Clone)]
pub struct PreparedHeadPreactivationRows(RetainedOwner<HeadPreactivationRowsInner>);

impl PreparedRootSession {
    /// Build all original first-head Gemm preactivation rows, then permit
    /// checked borrowed access to requested original output indices.
    ///
    /// The actual wrapper/control/alignment bound and bounded fixed work are
    /// admitted before the original projection and row builders. Those builders
    /// retain their existing complete allocation, work, FP, geometry and policy
    /// checks. Every claim spends this source's original shared budget and its
    /// inherited whole-source cap; this is not an allowance renewal or an exact
    /// byte-cap acceptance promise. Failure returns no partial public owner.
    ///
    /// This rows-only operation requests no optional clipping, retention or
    /// backward-intermediate product, and does not complete consumer requests
    /// made to another proof/evaluation operation. Those requests are not
    /// persistent session settings. The actual source precision policy passes
    /// unchanged to the original projector and all its checks remain intact.
    /// No construction flag or requested root precision is replaced.
    pub fn head_preactivation_rows(
        &self,
        requested: Option<Instant>,
    ) -> Result<PreparedHeadPreactivationRows> {
        let mut at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let mut lease = self
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: owner_bytes()?,
                    scratch_bytes: 0,
                    work: OWNER_WORK,
                    proposals: 0,
                },
            )
            .map_err(refusal)?;
        at = self.deadline(Some(lease.deadline().map_or(at, |d| d.min(at))))?;
        let adapter = project_prepared_root(
            self.source().clone(),
            self.source().construction().prefix().root().policy(),
            BackwardIntermRuntimePolicy::default(),
            false,
            None,
            Some(at),
        )?;
        at = at.min(adapter.deadline(Some(at))?);
        let rows = build_prepared_ordinary_rows(adapter.clone(), Some(at))?;
        at = at.min(rows.deadline(Some(at))?);
        rows.validate_adapter(&adapter, Some(at))?;
        drop(adapter);
        self.checkpoint(Some(at))?;
        lease.mark_materialized().map_err(refusal)?;
        let result = PreparedHeadPreactivationRows(
            lease
                .commit_retained_with(|| HeadPreactivationRowsInner { rows, deadline: at })
                .map_err(refusal)?,
        );
        result.checkpoint(Some(at))?;
        Ok(result)
    }
}

impl PreparedHeadPreactivationRows {
    /// Check original numeric source/domain, ledger, live FP qualification,
    /// cancellation and the earliest inherited clock before borrowing rows.
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.0.rows.checkpoint(Some(at))?;
        if !self
            .0
            .shares_ledger(self.0.rows.adapter().source().budget())
            || self.0.bytes() != owner_bytes()?
        {
            return Err(invalid("head preactivation reader lost its original owner"));
        }
        Ok(())
    }

    /// Minimum captured session/request/header/row/source/model deadline,
    /// including the original model's current instance-budget restriction.
    pub fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0.rows.deadline(Some(
            requested.map_or(self.0.deadline, |at| at.min(self.0.deadline)),
        ))
    }

    /// Check the exact original source/domain and ledger, respecting both
    /// owners' clocks. Different domains of one compiled model cannot pair.
    pub fn validate_session(
        &self,
        session: &PreparedRootSession,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(Some(session.deadline(requested)?))?;
        session.checkpoint(Some(at))?;
        self.checkpoint(Some(at))?;
        self.0
            .rows
            .adapter()
            .validate_source(session.source(), Some(at))
    }

    /// Whether both readers share the exact same immutable row payload owner.
    /// Separate calls may share a source but build independent paid row owners.
    pub fn same_owner(&self, other: &Self) -> bool {
        self.0.rows.same_owner(&other.0.rows)
    }

    /// Historical original input-coordinate count.
    pub fn n_in(&self) -> usize {
        self.0.rows.n_in()
    }

    /// Historical first-head preactivation count, not final output classes.
    pub fn columns(&self) -> usize {
        self.0.rows.columns()
    }

    /// Historical original operation ID of the first head Gemm. Its output
    /// tensor has original tensor ID `first_head_operation() + 1`.
    pub fn first_head_operation(&self) -> usize {
        self.0.rows.adapter().net().i_gemm1
    }

    /// Original input box on which these affine enclosures apply. Borrowing
    /// this numeric snapshot does not extend time or later work authority.
    pub fn declared_domain(&self, requested: Option<Instant>) -> Result<(&[f64], &[f64])> {
        self.checkpoint(requested)?;
        let root = self
            .0
            .rows
            .adapter()
            .source()
            .construction()
            .prefix()
            .root();
        Ok((root.lo(), root.hi()))
    }

    /// Lower affine snapshot for the requested original first-head output.
    /// Out-of-range IDs are refused. This borrows existing planes without a
    /// row clone, copy, new allocation or caller-supplied callback.
    pub fn lower(
        &self,
        original_output: usize,
        requested: Option<Instant>,
    ) -> Result<PreparedHeadPreactivationRow<'_>> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        Ok(PreparedHeadPreactivationRow(
            self.0.rows.lower(original_output, Some(at))?,
        ))
    }

    /// Upper affine snapshot for the requested original first-head output.
    pub fn upper(
        &self,
        original_output: usize,
        requested: Option<Instant>,
    ) -> Result<PreparedHeadPreactivationRow<'_>> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        Ok(PreparedHeadPreactivationRow(
            self.0.rows.upper(original_output, Some(at))?,
        ))
    }
}

/// A lifetime-bound read-only affine snapshot of one original preactivation.
///
/// Let `v(x) = bias + sum(a_i * x_i)` and
/// `eta(x) = bias_error + sum(error_i * abs(x_i))` using the original input IDs
/// from `entries()`. The lower snapshot represents `v(x) - eta(x)`; the upper
/// represents `v(x) + eta(x)` on the owner's original declared input box.
/// Errors are nonnegative and the independent bias/error must both be kept.
/// A row is not a completed concretized bound. These expressions describe
/// producer data, not a floating-point evaluation recipe: any downstream
/// evaluation needs its own correct outward arithmetic.
///
/// A snapshot borrow keeps the paid original owner alive. Reading its copied
/// fields or iterating it after a later expiry/cancellation does not grant
/// ongoing source, numerical work, proof or publication authority. Obtain a
/// fresh checked borrow before starting another operation under that owner.
pub struct PreparedHeadPreactivationRow<'a>(PreparedOrdinaryRowView<'a>);

impl PreparedHeadPreactivationRow<'_> {
    /// Original first-head Gemm output index, never a final output-class ID.
    pub fn original_output(&self) -> usize {
        self.0.original_output()
    }

    /// Independent nominal affine constant; do not fold it into coefficients.
    pub fn bias(&self) -> f64 {
        self.0.bias()
    }

    /// Nonnegative independent constant error, charged separately from bias.
    pub fn bias_error(&self) -> f64 {
        self.0.bias_error()
    }

    /// Ordered `(original_input, coefficient, nonnegative_error)` values.
    /// This iterator borrows existing planes and allocates nothing. Consumers
    /// own the cost and outward correctness of any arithmetic over these data.
    pub fn entries(&self) -> impl Iterator<Item = (usize, f64, f64)> + '_ {
        self.0.entries()
    }
}

#[cfg(test)]
#[path = "head_preactivation_rows_tests.rs"]
mod tests;
