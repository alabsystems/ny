// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit, paid access to the retained original input-row selection.
//!
//! A session must have requested and completed input-row retention during
//! construction. This API does not rebuild missing rows, enable retention, or
//! change the default root route. The opaque owners keep the original source
//! and its ledger alive; cloning shares their existing reservations.
//!
//! `PreparedInputSplit` and `PreparedInputSplitSide` are the original plain
//! request types, publicly named so a caller can submit its own borrowed slice
//! without an unpriced conversion allocation. For example, after selecting
//! an original `(layer, target)` from `rows.ids(None)?`, construct
//! `PreparedInputSplit { layer, target, side: PreparedInputSplitSide::Active }`
//! and pass `&[request]` to `rows.necessary_halfspaces(&[request], None)?`.
//! This declares a premise; it does not establish that the phase holds.
//!
//! Halfspaces are necessary conditions on the original declared input box for
//! the requested closed phases (zero belongs to both phases). They do not
//! establish phase feasibility, partition coverage, a proof, or publication
//! authority. Any later dot/dual arithmetic needs its own outward consumer.

use std::time::Instant;

use ny_core::Result;

use super::cli_proof::PreparedRootSession;
use super::retained::{
    build_prepared_clip_rows, PreparedClipRows, PreparedHalfspaceBatch, PreparedSplit,
};
use crate::margin_row::clip::Halfspace;

/// An immutable, paid selection of genuine original retained input rows.
///
/// All IDs use original trunk-layer and preactivation indices. No raw row
/// planes or mutable owner state are exposed. The source and reservations stay
/// alive until the last dependent owner is dropped, including after session
/// drop. A content fingerprint is a cache hint, never source authority.
#[derive(Clone)]
pub struct PreparedInputRows(PreparedClipRows);

impl PreparedRootSession {
    /// Select every final unstable original input row delivered by this
    /// session's completed construction. Missing retention is a refusal even
    /// when there would be no final unstable rows.
    ///
    /// Metadata is admitted on the original shared budget before allocation.
    /// The effective deadline includes this session's restricted deadline;
    /// later requests cannot extend it. Construction flags are unchanged.
    ///
    /// Genuine session construction already atomically restricts the original
    /// shared query ledger to the whole-source byte ceiling. This selection and
    /// its descendants inherit that ceiling, including any smaller caller
    /// restriction. They cannot renew or loosen the original shared allowance.
    pub fn retained_input_rows(&self, requested: Option<Instant>) -> Result<PreparedInputRows> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        Ok(PreparedInputRows(build_prepared_clip_rows(
            self.source(),
            Some(at),
        )?))
    }
}

impl PreparedInputRows {
    /// Check the retained source, original ledger, cancellation and deadline.
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        self.0.validate_source(self.0.source(), requested)
    }

    /// The minimum of the requested, inherited and original source deadlines.
    pub fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0.deadline(requested)
    }

    /// Check actual source/ledger identity against a session, also respecting
    /// that session's restricted clock. Equal content or limits cannot suffice.
    pub fn validate_session(
        &self,
        session: &PreparedRootSession,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(Some(session.deadline(requested)?))?;
        session.checkpoint(Some(at))?;
        self.0.validate_source(session.source(), Some(at))
    }

    /// Whether both values share this exact selection owner and reservation.
    pub fn same_owner(&self, other: &Self) -> bool {
        self.0.same_owner(&other.0)
    }

    /// Historical selection size; this metadata does not extend its clock.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the selection contains no final unstable original rows.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Content/selection cache hint. Matching bytes do not grant provenance.
    pub fn content_fingerprint(&self) -> &[u8; 32] {
        self.0.digest()
    }

    /// This selection's metadata reservation, excluding independently owned
    /// source and row storage. This is neither total source memory nor RSS.
    pub fn resident_bytes(&self) -> usize {
        self.0.resident_bytes()
    }

    /// Check the source/clock/ledger, then iterate ordered original IDs without
    /// allocating. Holding these metadata IDs does not extend source authority.
    pub fn ids(
        &self,
        requested: Option<Instant>,
    ) -> Result<impl Iterator<Item = (usize, usize)> + '_> {
        self.checkpoint(requested)?;
        Ok(self.0.ids())
    }

    /// Admit an immutable ordered unique subsequence on the original ledger.
    /// Missing, duplicate or out-of-order original IDs are refused. The source
    /// and row planes remain shared; only the existing priced metadata is copied.
    pub fn filter(
        &self,
        ordered_ids: &[(usize, usize)],
        requested: Option<Instant>,
    ) -> Result<Self> {
        Ok(Self(self.0.filter(ordered_ids, requested)?))
    }

    /// Admit necessary halfspaces for declared original phases in request
    /// order. The request slice is borrowed directly; the original builder
    /// prices its stored requests, coefficients, errors and owner before allocation.
    ///
    /// Every ID must belong to this selection. Requests do not fix phases in
    /// the source, and repeated or opposing premises do not certify feasibility.
    pub fn necessary_halfspaces(
        &self,
        requests: &[PreparedSplit],
        requested: Option<Instant>,
    ) -> Result<PreparedInputHalfspaceBatch> {
        Ok(PreparedInputHalfspaceBatch(
            self.0.halfspaces(requests, requested)?,
        ))
    }
}

/// A paid immutable batch of necessary input halfspaces `a · x + d <= 0`.
///
/// Each halfspace already includes the original producer's charged pointwise
/// error on its original declared input box. Numerical slices are available
/// only after a source/clock/ledger checkpoint. A returned shared borrow keeps
/// this owner alive but is a numerical snapshot, not an ongoing authorization
/// to perform later proof arithmetic after expiry or cancellation.
#[derive(Clone)]
pub struct PreparedInputHalfspaceBatch(PreparedHalfspaceBatch);

impl PreparedInputHalfspaceBatch {
    /// Check original source/ledger ownership, cancellation and deadline.
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        self.0.validate_source(self.0.source(), requested)
    }

    /// The minimum of the requested, inherited and original source deadlines.
    pub fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0.deadline(requested)
    }

    /// Check actual source/ledger identity against a session and its clock.
    pub fn validate_session(
        &self,
        session: &PreparedRootSession,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(Some(session.deadline(requested)?))?;
        session.checkpoint(Some(at))?;
        self.0.validate_source(session.source(), Some(at))
    }

    /// Whether both values share this exact batch owner and reservation.
    pub fn same_owner(&self, other: &Self) -> bool {
        self.0.same_owner(&other.0)
    }

    /// Selection-content cache hint for the retained input-row owner used to
    /// build this batch. This is not the raw model/source digest or authority.
    pub fn selection_fingerprint(&self) -> &[u8; 32] {
        self.0.source_digest()
    }

    /// Batch-content cache hint, including request order and numerical data.
    pub fn content_fingerprint(&self) -> &[u8; 32] {
        self.0.digest()
    }

    /// This batch's existing reservation, excluding independently retained
    /// source/row owners. This does not report total source memory or RSS.
    pub fn resident_bytes(&self) -> usize {
        self.0.resident_bytes()
    }

    /// Original phase declarations in the same order as numerical outputs.
    pub fn splits(&self, requested: Option<Instant>) -> Result<&[PreparedSplit]> {
        self.checkpoint(requested)?;
        Ok(self.0.splits())
    }

    /// Read-only necessary constraints. No raw plane or mutating API is exposed.
    pub fn halfspaces(&self, requested: Option<Instant>) -> Result<&[Halfspace]> {
        self.checkpoint(requested)?;
        Ok(self.0.halfspaces())
    }

    /// Producer pointwise errors already charged into the corresponding
    /// halfspace constants. Consumers must not subtract these a second time.
    pub fn charged_errors(&self, requested: Option<Instant>) -> Result<&[f64]> {
        self.checkpoint(requested)?;
        Ok(self.0.charged_errors())
    }
}

#[cfg(test)]
#[path = "retained_input_rows_tests.rs"]
mod tests;
