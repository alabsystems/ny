// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! The global SDP is an additional checked row, never an AY status.
use super::*;

/// Preserve every old plane and its actual attaining LP. Optional global work
/// shares the original absolute stage and predetermined continuation reserve.
#[must_use]
pub fn propose_root_mip_cuts_with_global_sdp(
    problem: &HullMarginProblem,
    deadline: Instant,
    continuation_reserve: Duration,
) -> Option<HullRootMipCuts<'_>> {
    let mut cuts = propose_root_mip_cuts_with_groups(problem, deadline, continuation_reserve)?;
    // The optional proposer accounts for its scheduling/floor check and skips
    // its optimizer when the retained old bound is already positive.
    cuts.global_sdp = sdp::propose(&cuts, cuts.deadline, continuation_reserve);
    live(cuts.deadline).then_some(cuts)
}

impl HullRootMipCuts<'_> {
    /// Original whole-model binding must succeed before this proof can be
    /// published. No later end renews the immutable original stage authority.
    pub fn compact_root_proof(&self, end: Instant) -> Option<HullAggregateRootProof> {
        if !self.compact_mode {
            return None;
        }
        let end = end.min(self.deadline);
        let end = self
            .compact
            .as_ref()
            .and_then(|r| r.group_budget.as_ref())
            .map_or(end, |b| end.min(b.authority));
        let end = ny_core::instance_budget::deadline().map_or(end, |d| d.min(end));
        let mut poll = || live(end);
        let row = compact::replay(self, end, &mut poll)?;
        let lower = compact::root_lower(self, &row, end, &mut poll)?;
        let proof = HullAggregateRootProof {
            lower,
            deadline: end,
            route: "compact",
        };
        proof.verified().then_some(proof)
    }

    /// A separate full-q replay of the selected global witness. Appending the
    /// original authenticated model is still required before publication.
    #[must_use]
    pub fn global_sdp_root_proof(&self, end: Instant) -> Option<HullAggregateRootProof> {
        let row = self.global_sdp.as_ref()?;
        let end = end.min(self.deadline).min(row.authority());
        let end = ny_core::instance_budget::deadline().map_or(end, |d| d.min(end));
        if !live(end) {
            return None;
        }
        require_hull_environment().ok()?;
        let checked = row.replay(self, end, &mut || live(end))?;
        let proof = HullAggregateRootProof {
            lower: checked.lower,
            deadline: end,
            route: "global-sdp",
        };
        proof.verified().then_some(proof)
    }

    /// Keep the best completed positive certificate and its opaque authority;
    /// a worse or refused optional global witness cannot replace an old proof.
    #[must_use]
    pub fn best_root_proof(&self, end: Instant) -> Option<HullAggregateRootProof> {
        let old = self.compact_root_proof(end);
        let global = self.global_sdp_root_proof(end);
        match (old, global) {
            (Some(a), Some(b)) => Some(if b.lower() > a.lower() { b } else { a }),
            (a, b) => a.or(b),
        }
    }
}
