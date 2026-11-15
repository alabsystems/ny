// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit, paid structural liveness for the complete original model DAG.
//!
//! This describes original tensor edges and last-use boundaries, including
//! head operations that root construction does not execute. It is not a trunk
//! execution trace, measured allocation size, RSS, or a numerical live-byte
//! estimate. Its IDs and counts do not establish a bound or proof authority.

use std::time::Instant;

use ny_core::Result;

use super::cli_proof::PreparedRootSession;
use super::compiler::OriginalModel;
use super::ir::{earliest, invalid, refusal};
use super::liveness::OriginalLiveness;
use crate::certify::work::{LeaseClass, RetainedOwner, WorkRequest};

// Fixed owner/checkpoint work only. OriginalLiveness separately admits its
// unchanged graph-sized work, persistent vectors and construction scratch.
const OWNER_WORK: u64 = 512;

struct ModelDagLivenessInner {
    model: OriginalModel,
    schedule: OriginalLiveness,
    deadline: Instant,
}

fn owner_bytes() -> Result<usize> {
    RetainedOwner::<ModelDagLivenessInner>::metadata_bytes().map_err(refusal)
}

/// A shared, paid original model-DAG release schedule.
///
/// Tensor zero is the original input; operation `op` produces tensor `op + 1`.
/// Consumer counts include each original edge, including both Add arms when
/// equal. Unused intermediate outputs release at their producer. The final
/// output releases only at the separate finish boundary after all operations.
///
/// Only an authentic completed session can create this owner. It retains the
/// compiled model, not root boxes, retained rows or the session's other numeric
/// products. Clones share the same reservations until their last drop. The
/// schedule is independent of a particular declared input domain.
#[derive(Clone)]
pub struct PreparedModelDagLiveness(RetainedOwner<ModelDagLivenessInner>);

impl PreparedRootSession {
    /// Admit the complete original model's structural release schedule.
    ///
    /// This opt-in diagnostic leaves construction and retention flags intact.
    /// Its actual owner/control/alignment bound is admitted before allocation;
    /// the original liveness builder separately admits its unchanged vectors,
    /// scratch and graph work. These conservative quotes overlap inline owner
    /// metadata and are not an exact allocation-price or byte-cap acceptance
    /// promise. All claims spend this session's original shared ledger and its
    /// already inherited whole-source ceiling; no allowance is renewed.
    pub fn model_dag_liveness(
        &self,
        requested: Option<Instant>,
    ) -> Result<PreparedModelDagLiveness> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let model = self.source().construction().prefix().root().model().clone();
        let mut lease = model
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
        let at = model
            .deadline(earliest(Some(at), lease.deadline()))?
            .ok_or_else(|| invalid("model DAG diagnostic lost its inherited deadline"))?;
        let schedule = OriginalLiveness::new(model.clone(), Some(at))?;
        if !schedule.same_model(&model) {
            return Err(invalid("model DAG diagnostic has a foreign model"));
        }
        let at = model
            .deadline(Some(at))?
            .ok_or_else(|| invalid("model DAG diagnostic lost its inherited deadline"))?;
        lease.mark_materialized().map_err(refusal)?;
        let result = PreparedModelDagLiveness(
            lease
                .commit_retained_with(|| ModelDagLivenessInner {
                    model,
                    schedule,
                    deadline: at,
                })
                .map_err(refusal)?,
        );
        result.checkpoint(Some(at))?;
        Ok(result)
    }
}

impl PreparedModelDagLiveness {
    /// Check the original model, shared ledger, cancellation and clock.
    /// Structural data confers no numerical or source-publication authority.
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        self.deadline(requested)?;
        if !self.0.shares_ledger(self.0.model.budget())
            || self.0.bytes() != owner_bytes()?
            || !self.0.schedule.same_model(&self.0.model)
        {
            return Err(invalid("model DAG diagnostic owner binding mismatch"));
        }
        Ok(())
    }

    /// Minimum captured session/header/model, requested and current instance
    /// deadline. A later request cannot extend the captured session clock.
    pub fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0
            .model
            .deadline(earliest(Some(self.0.deadline), requested))?
            .ok_or_else(|| invalid("model DAG diagnostic lost its inherited deadline"))
    }

    /// Pair original IDs with this exact compiled model and ledger. Separate
    /// input domains prepared from one compilation may share this schedule;
    /// independently compiled equal content or equal limits cannot substitute.
    pub fn validate_session(
        &self,
        session: &PreparedRootSession,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(Some(session.deadline(requested)?))?;
        session.checkpoint(Some(at))?;
        self.checkpoint(Some(at))?;
        let model = session.source().construction().prefix().root().model();
        if !self.0.model.same_model(model) || !self.0.model.budget().shares_ledger(session.budget())
        {
            return Err(invalid(
                "model DAG diagnostic requires its original compiled model",
            ));
        }
        Ok(())
    }

    /// Historical whole-model operation count, including head operations.
    pub fn operation_count(&self) -> usize {
        self.0.model.net().ops.len()
    }

    /// Historical original tensor count, including the original input.
    pub fn tensor_count(&self) -> usize {
        self.0.model.net().tsize.len()
    }

    /// Check the original clock/ledger, then return this tensor's edge count.
    /// An out-of-range original tensor ID is an error, not a zero-use tensor.
    pub fn consumer_count(&self, tensor: usize, requested: Option<Instant>) -> Result<usize> {
        self.checkpoint(requested)?;
        self.0
            .schedule
            .consumer_count(tensor)
            .ok_or_else(|| invalid("model DAG diagnostic tensor is out of range"))
    }

    /// Original tensor IDs released after this whole-model operation. The
    /// ordered borrowed metadata allocates nothing and is a static snapshot;
    /// retaining a borrow does not extend time or grant later proof work.
    pub fn release_after(&self, op: usize, requested: Option<Instant>) -> Result<&[usize]> {
        self.checkpoint(requested)?;
        self.0
            .schedule
            .release_after(op)
            .ok_or_else(|| invalid("model DAG diagnostic operation is out of range"))
    }

    /// Final-output releases after the complete original DAG. This separate
    /// boundary must not replace a numerical executor's own trunk cleanup.
    pub fn finish_releases(&self, requested: Option<Instant>) -> Result<&[usize]> {
        self.checkpoint(requested)?;
        Ok(self.0.schedule.finish_releases())
    }
}

#[cfg(test)]
#[path = "model_dag_liveness_tests.rs"]
mod tests;
