// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #graph-bab-cover-ledger: the graph BaB drivers' side of the completeness
//! ledger (`certify::graph`).
//!
//! A ledgered driver ARMS a frame on its verifier's `graph_cover` slot at
//! entry and SETTLES it at exit ([`BetaCrownVerifier::settle_graph_cover`]).
//! In between, every site that CLOSES a region — a domain or child whose
//! certified bound clears the threshold, a child or split half-space whose
//! region the lane's certified node bounds show empty, an input box a clip
//! closed or shrank — records the fact through the `cover_note_*` helpers
//! below. Each helper is one relaxed atomic load when no frame is armed, so
//! the shared batch processors cost nothing on unledgered routes.
//!
//! Settling is the lane's own `Verified` put to the test:
//!
//! * proved — the verdict stands; the cover is captured for publication and
//!   `[cover-gate] graph-frame lane=... outcome=proved ...` is printed;
//! * unbuilt — the verdict stands on the counted legacy door
//!   (`outcome=unbuilt reason=...`);
//! * refused — the lane's certified leaves do not cover its root: `Verified`
//!   is WITHDRAWN (`Unknown`) and `[cover-gate] REFUSED scope=invocation ...`
//!   is printed. That line on a banked row is a latent soundness finding.

use std::time::Instant;

use ny_tensor::BoundedTensor;

use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::beta_crown::state::GraphBetaState;
use crate::certify::graph::{GraphCoverScope, GraphCoverVerdict, GraphRecordKey};

use super::super::BetaCrownVerifier;

impl BetaCrownVerifier {
    fn cover_key<'a>(&self, objective: &'a [f32], threshold: f32) -> GraphRecordKey<'a> {
        GraphRecordKey {
            objective,
            threshold,
            verify_upper: self.config.verify_upper_bound,
        }
    }

    /// A domain (split history `history`) closed by the bound pair the lane's
    /// closure rule (`domain_is_verified`) accepted. `beta` is the domain's
    /// β state: B2.2 streams it as the leaf's dual-point HINT under the Proof
    /// profile (never read otherwise; a reference costs nothing).
    pub(crate) fn cover_note_bound(
        &self,
        objective: &[f32],
        threshold: f32,
        history: &GraphSplitHistory,
        beta: Option<&GraphBetaState>,
        lower: f32,
        upper: f32,
    ) {
        if self.graph_cover.is_armed() {
            self.graph_cover.note_bound(
                self.cover_key(objective, threshold),
                history,
                beta,
                lower,
                upper,
            );
        }
    }

    /// A region the lane's certified node bounds show EMPTY: the domain with
    /// history `history`, narrowed by `extra` when the fact is about a child
    /// half-space the lane never built (`with_constraint` → `Ok(None)` or an
    /// `InfeasibleDomain` error).
    pub(crate) fn cover_note_infeasible(
        &self,
        objective: &[f32],
        threshold: f32,
        history: &GraphSplitHistory,
        extra: Option<&GraphNeuronConstraint>,
    ) {
        if self.graph_cover.is_armed() {
            self.graph_cover
                .note_infeasible(self.cover_key(objective, threshold), history, extra);
        }
    }

    /// A close the ledger cannot express: the armed frame (if any) becomes
    /// UNBUILT, so the lane's verdict publishes through the counted legacy
    /// door — never refused on a ledger limitation.
    pub(crate) fn cover_note_unbuildable(&self, reason: &'static str) {
        if self.graph_cover.is_armed() {
            self.graph_cover.note_unbuildable(reason);
        }
    }

    /// An input box closed by the bound pair the lane's closure rule accepted.
    pub(crate) fn cover_note_box_bound(
        &self,
        objective: &[f32],
        threshold: f32,
        input: &BoundedTensor,
        lower: f32,
        upper: f32,
    ) {
        if self.graph_cover.is_armed() {
            self.graph_cover.note_box_bound(
                self.cover_key(objective, threshold),
                input,
                lower,
                upper,
            );
        }
    }

    /// An input box a clip closed whole (the clip verified it).
    pub(crate) fn cover_note_box_clip_closed(
        &self,
        objective: &[f32],
        threshold: f32,
        input: &BoundedTensor,
    ) {
        if self.graph_cover.is_armed() {
            self.graph_cover
                .note_box_clip_closed(self.cover_key(objective, threshold), input);
        }
    }

    /// An input box the lane's batched IBP prescreen verified.
    pub(crate) fn cover_note_box_screened(
        &self,
        objective: &[f32],
        threshold: f32,
        input: &BoundedTensor,
    ) {
        if self.graph_cover.is_armed() {
            self.graph_cover
                .note_box_screened(self.cover_key(objective, threshold), input);
        }
    }

    /// An input clip of `pre` to `post` (the excluded part is closed).
    pub(crate) fn cover_note_box_clip(
        &self,
        objective: &[f32],
        threshold: f32,
        pre: &BoundedTensor,
        post: &BoundedTensor,
    ) {
        if self.graph_cover.is_armed() {
            self.graph_cover
                .note_box_clip(self.cover_key(objective, threshold), pre, post);
        }
    }

    /// Settle a ledgered driver's frame against its result (see the module
    /// docs). The cover check never runs past `deadline` (the driver's own
    /// boundary, else the verifier's), nor past the ledger's own wall cap.
    pub(crate) fn settle_graph_cover(
        &self,
        lane: &'static str,
        scope: GraphCoverScope<'_>,
        mut result: BetaCrownResult,
        deadline: Option<Instant>,
    ) -> BetaCrownResult {
        let verified = matches!(result.result, BabVerificationStatus::Verified);
        let deadline = deadline.or(self.config.alpha_config.deadline);
        match scope.finish(verified, deadline) {
            GraphCoverVerdict::NotArmed => {}
            GraphCoverVerdict::Proved(summary) => eprintln!(
                "[cover-gate] graph-frame lane={lane} outcome=proved leaves={} lane_leaves={} \
                 max_depth={} min_margin={:e} records={}",
                summary.leaves,
                summary.lane_leaves,
                summary.max_depth,
                summary.min_margin,
                summary.records,
            ),
            GraphCoverVerdict::Unbuilt(reason) => {
                eprintln!("[cover-gate] graph-frame lane={lane} outcome=unbuilt reason={reason}");
            }
            GraphCoverVerdict::Refused(refusal) => {
                eprintln!(
                    "[cover-gate] REFUSED scope=invocation lane={lane} reason={refusal} \
                     (the lane's certified leaves do not cover its root: Verified withdrawn)"
                );
                result.result = BabVerificationStatus::Unknown {
                    reason: format!("graph cover gate refused ({lane}): {refusal}"),
                };
            }
        }
        result
    }
}
