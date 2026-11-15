// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit, paid original-layer rows without historical all-row retention.
//!
//! A completed session keeps an authentic prefix before every original trunk
//! ReLU even when construction discarded its row planes. This operation pays
//! for one requested dense tile on that prefix. It does not deliver historical
//! retention, alter construction receipts, or acquire legacy root authority.

use std::time::Instant;

use ny_core::Result;

use super::cli_proof::PreparedRootSession;
use super::ir::{invalid, refusal};
use super::prefix::PreparedPrefix;
use super::publication::PreparedRootSource;
use super::PreparedInputSplitSide;
use crate::certify::work::{LeaseClass, RetainedOwner, WorkRequest};
use crate::margin_row::engine::run_prepared_prefix_input_rows;
use crate::margin_row::input_clipping::{input_halfspace_prepared, ScopedInputHalfspace};
use crate::margin_row::input_rows::{InputRowPair, InputRowView};

const CONTROL_WORK: u64 = 1024;

struct SparseRowsInner {
    source: PreparedRootSource,
    rows: InputRowPair<'static>,
    layer: usize,
    op: usize,
    deadline: Instant,
}

fn row_owner_bytes() -> Result<usize> {
    RetainedOwner::<SparseRowsInner>::metadata_bytes().map_err(refusal)
}

fn row_owner_work(targets: usize) -> Result<u64> {
    u64::try_from(targets)
        .ok()
        .and_then(|n| n.checked_mul(32))
        .and_then(|n| n.checked_add(CONTROL_WORK))
        .ok_or_else(|| invalid("sparse input-row control work overflow"))
}

/// Shared original-layer preactivation sandwiches on the declared input box.
///
/// Request order is retained, including stable original neurons. Columns are
/// positions in that request, not compacted or sorted original IDs. Each call
/// builds its own paid dense F64 rows; clones share this owner and its claims.
/// A historical `row_f32` construction request does not describe this product.
/// Missing all-row retention remains a refusal in the existing retention APIs.
///
/// These rows carry numerical premises, not phase truth, whole-property
/// coverage, a proof grade, source-file authentication or publication authority.
/// Downstream arithmetic/scans/copies need their own budget and correctness.
#[derive(Clone)]
pub struct PreparedSparseInputRows(RetainedOwner<SparseRowsInner>);

impl PreparedRootSession {
    /// Pay for requested original preactivation IDs in one original trunk layer.
    ///
    /// The unchanged producer accepts a nonempty tile of at most 64 unique
    /// in-range targets and retains its existing CPU/geometry/policy governors.
    /// This entry neither tiles nor changes route on a refusal. Its actual
    /// shared header and ordered-handoff work are admitted before production;
    /// the producer separately admits target validation/copy, rows and scratch.
    /// All claims overlap on the original shared, already capped ledger.
    /// Additional headers are not an exact old-cap acceptance promise or RSS.
    pub fn sparse_input_rows(
        &self,
        original_layer: usize,
        targets: &[usize],
        requested: Option<Instant>,
    ) -> Result<PreparedSparseInputRows> {
        let mut at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let mut header = self
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: row_owner_bytes()?,
                    work: row_owner_work(targets.len())?,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        at = self.deadline(Some(header.deadline().map_or(at, |d| d.min(at))))?;
        let construction = self.source().construction();
        let product = construction
            .layer_products(original_layer)
            .ok_or_else(|| invalid("sparse input rows missing original layer"))?;
        let prefix = product.prefix_before();
        let op = product.op();
        if product.layer() != original_layer
            || prefix.layer_count() != original_layer
            || !prefix.root().same_root(construction.prefix().root())
        {
            return Err(invalid("sparse input rows changed original layer/prefix"));
        }
        prefix.for_relu(op)?;
        // The original producer admits T*T+16*T validation work and the real
        // target owner before scanning/copying this untrusted borrowed slice.
        let rows = run_prepared_prefix_input_rows(prefix, targets, Some(at))?;
        at = self.deadline(Some(at.min(rows.deadline())))?;
        // This public handoff compares the independent caller request with
        // actual delivered order; its entire scan was paid above at 32*T.
        rows.validate_prepared_prefix_with_deadline(prefix, op, targets, at)?;
        self.checkpoint(Some(at))?;
        header.mark_materialized().map_err(refusal)?;
        let result = PreparedSparseInputRows(
            header
                .commit_retained_with(|| SparseRowsInner {
                    source: self.source().clone(),
                    rows,
                    layer: original_layer,
                    op,
                    deadline: at,
                })
                .map_err(refusal)?,
        );
        result.checkpoint(Some(at))?;
        Ok(result)
    }
}

impl PreparedSparseInputRows {
    fn prefix(&self) -> Result<&PreparedPrefix> {
        let construction = self.0.source.construction();
        let product = construction
            .layer_products(self.0.layer)
            .ok_or_else(|| invalid("sparse input rows lost original layer"))?;
        let prefix = product.prefix_before();
        if product.layer() != self.0.layer
            || product.op() != self.0.op
            || prefix.layer_count() != self.0.layer
            || !prefix.root().same_root(construction.prefix().root())
        {
            return Err(invalid("sparse input rows changed source/prefix binding"));
        }
        Ok(prefix)
    }

    /// Minimum captured session/header/producer/source/request and live model
    /// instance deadline. Later requests cannot extend this owner's clock.
    pub fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0.source.deadline(Some(
            requested
                .map_or(self.0.deadline, |d| d.min(self.0.deadline))
                .min(self.0.rows.deadline()),
        ))
    }

    /// Check the actual completed source, live FP, current global clock,
    /// cancellation, original ledger, paid metadata and exact producer scope.
    /// The row pair's lightweight checkpoint alone is not source authority.
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.0.source.checkpoint(Some(at))?;
        if !self.0.shares_ledger(self.0.source.budget()) || self.0.bytes() != row_owner_bytes()? {
            return Err(invalid("sparse input rows lost original header claim"));
        }
        self.0.rows.validate_prepared_prefix_binding(
            self.prefix()?,
            self.0.op,
            self.0.rows.columns(),
            at,
        )
    }

    /// Exact completed source/domain identity, not model/digest/cap equality.
    pub fn validate_session(
        &self,
        session: &PreparedRootSession,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(Some(session.deadline(requested)?))?;
        session.checkpoint(Some(at))?;
        self.checkpoint(Some(at))?;
        if !self.0.source.same_source(session.source()) {
            return Err(invalid("sparse input rows foreign completed source"));
        }
        Ok(())
    }

    /// Historical original trunk-layer index.
    pub fn original_layer(&self) -> usize {
        self.0.layer
    }

    /// Historical original input-coordinate count.
    pub fn n_in(&self) -> usize {
        self.0.rows.n_in()
    }

    /// Historical requested column count, not complete unstable-row coverage.
    pub fn columns(&self) -> usize {
        self.0.rows.columns()
    }

    /// Producer-owned original IDs in exact caller order, without a new scan.
    pub fn targets(&self, requested: Option<Instant>) -> Result<&[usize]> {
        self.checkpoint(requested)?;
        Ok(self.0.rows.targets())
    }

    /// Original declared domain of these preactivation sandwiches.
    pub fn declared_domain(&self, requested: Option<Instant>) -> Result<(&[f64], &[f64])> {
        self.checkpoint(requested)?;
        let root = self.prefix()?.root();
        Ok((root.lo(), root.hi()))
    }

    /// Lower snapshot for a column POSITION in the original ordered request.
    pub fn lower(
        &self,
        column: usize,
        requested: Option<Instant>,
    ) -> Result<PreparedSparseInputRow<'_>> {
        self.checkpoint(requested)?;
        let target = self
            .0
            .rows
            .target(column)
            .ok_or_else(|| invalid("sparse input rows missing requested column"))?;
        let view = self
            .0
            .rows
            .lower(column)
            .ok_or_else(|| invalid("sparse input rows missing lower snapshot"))?;
        Ok(PreparedSparseInputRow { view, target })
    }

    /// Upper snapshot for a column POSITION in the original ordered request.
    pub fn upper(
        &self,
        column: usize,
        requested: Option<Instant>,
    ) -> Result<PreparedSparseInputRow<'_>> {
        self.checkpoint(requested)?;
        let target = self
            .0
            .rows
            .target(column)
            .ok_or_else(|| invalid("sparse input rows missing requested column"))?;
        let view = self
            .0
            .rows
            .upper(column)
            .ok_or_else(|| invalid("sparse input rows missing upper snapshot"))?;
        Ok(PreparedSparseInputRow { view, target })
    }

    /// Pay for one necessary condition for the requested closed phase.
    /// Zero belongs to both sides. This declares a premise, not its feasibility.
    /// The result borrows this external row owner; it contains no self-reference.
    pub fn necessary_halfspace(
        &self,
        column: usize,
        side: PreparedInputSplitSide,
        requested: Option<Instant>,
    ) -> Result<PreparedSparseInputHalfspace<'_>> {
        let mut at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let mut header = self
            .0
            .source
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: halfspace_owner_bytes()?,
                    work: CONTROL_WORK,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        at = self.deadline(Some(header.deadline().map_or(at, |d| d.min(at))))?;
        let halfspace = input_halfspace_prepared(
            &self.0.rows,
            self.prefix()?,
            self.0.op,
            self.0.rows.targets(),
            column,
            side,
            at,
        )?;
        at = self.deadline(Some(at.min(halfspace.deadline())))?;
        self.checkpoint(Some(at))?;
        header.mark_materialized().map_err(refusal)?;
        let result = PreparedSparseInputHalfspace(
            header
                .commit_retained_with(|| SparseHalfspaceInner {
                    rows: self,
                    halfspace,
                    deadline: at,
                })
                .map_err(refusal)?,
        );
        result.checkpoint(Some(at))?;
        Ok(result)
    }
}

/// Lifetime-bound, read-only original preactivation affine data.
///
/// With nominal v(x)=bias+sum(a_i*x_i) and nonnegative error
/// eta(x)=bias_error+sum(e_i*abs(x_i)), a lower view represents v-eta
/// and an upper view v+eta on the owner's declared domain. These expressions
/// describe producer data, not a floating evaluation recipe or a proof.
/// A fresh checked borrow is needed for a fresh operation after time/FP or
/// cancellation changes; existing snapshots confer no ongoing work authority.
pub struct PreparedSparseInputRow<'a> {
    view: InputRowView<'a>,
    target: usize,
}

impl PreparedSparseInputRow<'_> {
    /// Original preactivation ID, distinct from the requested column position.
    pub fn original_target(&self) -> usize {
        self.target
    }
    /// Independent nominal affine constant.
    pub fn bias(&self) -> f64 {
        self.view.bias()
    }
    /// Nonnegative independent constant error; keep it separate from bias.
    pub fn bias_error(&self) -> f64 {
        self.view.bias_error()
    }
    /// Borrow original-input-indexed coefficients and nonnegative errors.
    pub fn entries(&self) -> impl Iterator<Item = (usize, f64, f64)> + '_ {
        self.view.entries()
    }
}

struct SparseHalfspaceInner<'rows> {
    rows: &'rows PreparedSparseInputRows,
    halfspace: ScopedInputHalfspace<'rows, 'static>,
    deadline: Instant,
}

fn halfspace_owner_bytes() -> Result<usize> {
    RetainedOwner::<SparseHalfspaceInner<'_>>::metadata_bytes().map_err(refusal)
}

/// An owned necessary-condition payload borrowing its external paid row owner.
///
/// The original helper's output reservation and this new shared header are
/// separate real claims on the same ledger. Neither extends the original
/// source/row clock. Coefficients, constant and charged producer error are
/// read-only snapshots; any subsequent dot/dual needs its own outward consumer.
/// No phase truth, feasibility, coverage or publication authority is supplied.
pub struct PreparedSparseInputHalfspace<'rows>(RetainedOwner<SparseHalfspaceInner<'rows>>);

impl PreparedSparseInputHalfspace<'_> {
    /// Minimum captured output/header/row/source/request and live model clock.
    pub fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0.rows.deadline(Some(
            requested
                .map_or(self.0.deadline, |d| d.min(self.0.deadline))
                .min(self.0.halfspace.deadline()),
        ))
    }

    /// Revalidate the live source and both genuine public owner claims.
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.0.rows.checkpoint(Some(at))?;
        if !self.0.shares_ledger(self.0.rows.0.source.budget())
            || self.0.bytes() != halfspace_owner_bytes()?
        {
            return Err(invalid("sparse input halfspace lost original header claim"));
        }
        Ok(())
    }

    /// Require this exact completed source/domain, respecting both clocks.
    pub fn validate_session(
        &self,
        session: &PreparedRootSession,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(Some(session.deadline(requested)?))?;
        self.0.rows.validate_session(session, Some(at))?;
        self.checkpoint(Some(at))
    }

    /// Historical original layer and requested original neuron provenance.
    pub fn original_layer(&self) -> usize {
        self.0.rows.original_layer()
    }
    /// Historical requested original neuron ID.
    pub fn original_target(&self) -> usize {
        self.0.halfspace.original_target()
    }
    /// Requested closed phase premise; this does not establish it holds.
    pub fn side(&self) -> PreparedInputSplitSide {
        self.0.halfspace.side()
    }

    /// Necessary constraint is sum(a_i*x_i)+constant <= 0 on the declared box.
    pub fn coefficients(&self, requested: Option<Instant>) -> Result<&[f64]> {
        self.checkpoint(requested)?;
        Ok(&self.0.halfspace.halfspace().a)
    }
    /// Independently downward-rounded constraint constant.
    pub fn constant(&self, requested: Option<Instant>) -> Result<f64> {
        self.checkpoint(requested)?;
        Ok(self.0.halfspace.halfspace().d)
    }
    /// Producer error charged on original declared input endpoints.
    pub fn charged_error(&self, requested: Option<Instant>) -> Result<f64> {
        self.checkpoint(requested)?;
        Ok(self.0.halfspace.charged_error())
    }
}

#[cfg(test)]
#[path = "sparse_input_rows_tests.rs"]
mod tests;
