// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit ranked nominal advice from authentic retained original input rows.
//!
//! The existing producer selects per-layer top slack rows under one global
//! advice-payload policy. This is distinct from selecting every final unstable
//! row with `retained_input_rows`. No consuming proof/evaluation path changes.

use std::time::Instant;

use ny_core::Result;

use super::cli_proof::{PreparedRootRetention, PreparedRootSession};
use super::ir::invalid;
use super::retained::{build_prepared_retained, PreparedRetainedLayer, PreparedRetainedRows};
use crate::margin_row::input_rows::InputRowView;
use crate::margin_row::root::RetainCfg;

/// A shared paid selection of original rows and heuristic nominal f32 advice.
///
/// Selection uses the existing original final-gate `c * (u - l)` slack policy:
/// descending score with original-ID ties, then ascending selected IDs. The
/// per-layer limit and global advice-byte budget are applied in original layer
/// order; this is not global highest-score allocation across layers.
///
/// Original coefficient/bias errors are omitted and nominal values round to
/// f32. These packs are heuristic ADVICE, not certified lower/upper enclosures,
/// halfspaces, phase feasibility, proof grades or publication authority.
///
/// The existing owner already pays its actual metadata, selected indices and
/// advice packs on the original shared ledger. This newtype adds no allocation.
/// Clones share that owner and keep the exact numeric source/domain alive until
/// last drop. Equal content, compiled models or resource limits cannot replace
/// the original source. Consumers pay for their own scans, copies and ranker work.
#[derive(Clone)]
pub struct PreparedRankedRowAdvice(PreparedRetainedRows);

impl PreparedRootSession {
    /// Build a ranked heuristic advice selection using the original producer.
    ///
    /// Construction must have requested and completed input-row retention.
    /// Missing delivery refuses even for a zero requested payload or a solved
    /// source. This operation does not enable retention or change any default.
    ///
    /// `retention.budget_bytes` limits only both nominal f32 advice packs. The
    /// producer separately admits indices, layer/owner metadata, scratch and
    /// work on the same already capped original ledger; source/rows remain
    /// independently owned. No allowance, proof product or optional consumer
    /// request is renewed. Existing precision, scope and resource checks remain.
    ///
    /// The session/request minimum is forwarded into the existing paid builder
    /// and persists in its owner. No new projector, ordinary rows, callback,
    /// advice conversion or additional enclosing heap owner is introduced.
    pub fn ranked_row_advice(
        &self,
        retention: PreparedRootRetention,
        requested: Option<Instant>,
    ) -> Result<PreparedRankedRowAdvice> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        Ok(PreparedRankedRowAdvice(build_prepared_retained(
            self.source(),
            RetainCfg {
                per_layer: retention.per_layer,
                budget_bytes: retention.budget_bytes,
            },
            Some(at),
        )?))
    }
}

impl PreparedRankedRowAdvice {
    /// Check original source/domain, ledger, FP qualification, clock and cancellation.
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        self.0.validate_source(self.0.source(), requested)
    }

    /// Earliest retained producer/session/request/source/model deadline,
    /// including the original model's current instance-budget restriction.
    pub fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0.deadline(requested)
    }

    /// Pair with this exact original source/domain and ledger under both clocks.
    /// Different domains of one compiled model or equal hashes cannot substitute.
    pub fn validate_session(
        &self,
        session: &PreparedRootSession,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(Some(session.deadline(requested)?))?;
        session.checkpoint(Some(at))?;
        self.0.validate_source(session.source(), Some(at))
    }

    /// Whether both values share the actual same immutable selection and claims.
    pub fn same_owner(&self, other: &Self) -> bool {
        self.0.same_owner(&other.0)
    }

    /// Historical request copied from the producer's unchanged policy fields.
    pub fn request(&self) -> PreparedRootRetention {
        let cfg = self.0.cfg();
        PreparedRootRetention {
            per_layer: cfg.per_layer,
            budget_bytes: cfg.budget_bytes,
        }
    }

    /// Historical original layer count, including empty/budget-exhausted layers.
    pub fn layer_count(&self) -> usize {
        self.0.layers().len()
    }

    /// Nominal f32 advice payload bytes only, bounded by the original request.
    /// Metadata, scratch and independently retained source/rows are excluded.
    pub fn payload_bytes(&self) -> usize {
        self.0.payload_bytes()
    }

    /// Existing retained-owner charged capacity, including its metadata and
    /// conservative allowance, excluding independently owned source/rows.
    /// This is not measured allocation/RSS or a byte-cap acceptance promise.
    pub fn resident_bytes(&self) -> usize {
        self.0.resident_bytes()
    }

    /// Content/request/advice cache hint; matching bytes grant no source authority.
    pub fn content_fingerprint(&self) -> &[u8; 32] {
        self.0.digest()
    }

    /// Check live source/FP/clock/ledger, then borrow one original layer in O(1).
    /// An absent original layer is an error, not an empty successful selection.
    pub fn layer(
        &self,
        original_layer: usize,
        requested: Option<Instant>,
    ) -> Result<PreparedRankedRowAdviceLayer<'_>> {
        self.checkpoint(requested)?;
        let layer = self
            .0
            .layers()
            .get(original_layer)
            .ok_or_else(|| invalid("ranked advice layer is outside the original source"))?;
        if layer.layer() != original_layer {
            return Err(invalid("ranked advice original layer identity changed"));
        }
        Ok(PreparedRankedRowAdviceLayer(layer))
    }

    /// Borrow an original lower affine row selected by this ranked owner.
    ///
    /// This reads the original retained coefficient and error planes, never
    /// reconstructs them from nominal f32 advice. The actual original-ID lookup
    /// is paid on the same ledger before searching. No planes are copied or
    /// rebuilt, and absent/unselected IDs refuse. The borrow keeps this exact
    /// source/domain owner alive; consumer arithmetic still needs its own
    /// admission and outward evaluation.
    pub fn original_lower_row(
        &self,
        original_layer: usize,
        original_target: usize,
        requested: Option<Instant>,
    ) -> Result<PreparedRankedInputRow<'_>> {
        let (rows, column) = self.0.row(original_layer, original_target, requested)?;
        let view = rows
            .lower(column)
            .ok_or_else(|| invalid("ranked original lower row absent"))?;
        Ok(PreparedRankedInputRow {
            view,
            layer: original_layer,
            target: original_target,
        })
    }

    /// Borrow the separately stored original upper affine row and errors.
    /// The same original selection, ledger and clock requirements apply as to
    /// `original_lower_row`; neither row is the nominal advice pack.
    pub fn original_upper_row(
        &self,
        original_layer: usize,
        original_target: usize,
        requested: Option<Instant>,
    ) -> Result<PreparedRankedInputRow<'_>> {
        let (rows, column) = self.0.row(original_layer, original_target, requested)?;
        let view = rows
            .upper(column)
            .ok_or_else(|| invalid("ranked original upper row absent"))?;
        Ok(PreparedRankedInputRow {
            view,
            layer: original_layer,
            target: original_target,
        })
    }
}

/// Read-only snapshot of an authentic ranked selection's original affine row.
///
/// The producer's nominal expression is `bias + sum(a_i*x_i)`, with independent
/// nonnegative error `bias_error + sum(e_i*abs(x_i))`. A lower row represents
/// nominal minus error; an upper row nominal plus error, on the original
/// declared input box. These are mathematical data, not a floating evaluation
/// recipe, phase premise, proof grade or publication authority. Fresh operations
/// after time, cancellation or FP changes need a fresh checked borrow. A held
/// numeric snapshot does not renew any work allowance or authorization.
pub struct PreparedRankedInputRow<'a> {
    view: InputRowView<'a>,
    layer: usize,
    target: usize,
}

impl PreparedRankedInputRow<'_> {
    pub fn original_layer(&self) -> usize {
        self.layer
    }
    pub fn original_target(&self) -> usize {
        self.target
    }
    /// Independent nominal affine constant, not an error-adjusted intercept.
    pub fn bias(&self) -> f64 {
        self.view.bias()
    }
    /// Nonnegative producer constant error, separate from the nominal constant.
    pub fn bias_error(&self) -> f64 {
        self.view.bias_error()
    }
    /// Original input indices, coefficients and nonnegative coefficient errors.
    pub fn entries(&self) -> impl Iterator<Item = (usize, f64, f64)> + '_ {
        self.view.entries()
    }
}

/// Lifetime-bound snapshot of one original layer's selected heuristic advice.
///
/// IDs are ascending original preactivation IDs after top-score selection.
/// Positions index that same original final gate's unstable-ID list; they are
/// neither compacted neuron IDs nor producer-tile columns. Each selected row
/// has original-input coefficient slots followed by its nominal bias, for a
/// total `row_width()` of `n_in + 1` values in each direction. The first
/// `row_width() - 1` slots are coefficients and the last is the bias.
///
/// Both packs are row-major in `ids()` order. They omit original coefficient
/// and bias errors and round to f32, so "lower" and "upper" name only their
/// original producer planes; neither pack is a certified bound or halfspace.
///
/// Borrowing keeps the actual paid owner alive. These are static numeric
/// snapshots: reads after later expiry/cancellation confer no future work,
/// source, proof or publication authority. Obtain a fresh checked layer borrow
/// before starting another operation, and pay for consumer scans/copies/work.
pub struct PreparedRankedRowAdviceLayer<'a>(&'a PreparedRetainedLayer);

impl PreparedRankedRowAdviceLayer<'_> {
    /// Original trunk-layer ID.
    pub fn original_layer(&self) -> usize {
        self.0.layer()
    }

    /// Final original unstable-neuron census before policy/budget selection.
    pub fn eligible(&self) -> usize {
        self.0.eligible()
    }

    /// Number of selected original rows represented in both advice packs.
    pub fn selected_count(&self) -> usize {
        self.0.ids().len()
    }

    /// Ascending selected original preactivation IDs, not descending score order.
    pub fn ids(&self) -> &[usize] {
        self.0.ids()
    }

    /// Matching positions in the original final gate's unstable-ID list.
    pub fn unstable_positions(&self) -> &[usize] {
        self.0.unstable_positions()
    }

    /// Original input-coordinate count plus one independent nominal bias slot.
    pub fn row_width(&self) -> usize {
        self.0.naug()
    }

    /// Heuristic nominal f32 features from the original lower row plane.
    /// Coefficients occupy the first width-1 slots; the final slot is bias.
    pub fn lower_advice(&self) -> &[f32] {
        self.0.lower_advice()
    }

    /// Heuristic nominal f32 features from the original upper row plane.
    pub fn upper_advice(&self) -> &[f32] {
        self.0.upper_advice()
    }

    /// Both nominal f32 packs' payload bytes, excluding metadata/source/rows.
    pub fn payload_bytes(&self) -> usize {
        self.0.bytes()
    }
}

#[cfg(test)]
#[path = "ranked_row_advice_tests.rs"]
mod tests;
