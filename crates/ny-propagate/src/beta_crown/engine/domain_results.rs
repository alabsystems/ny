// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Result enums for parallel graph domain processing.

use crate::beta_crown::domain::{GraphBabDomain, MultiObjectiveGraphBabDomain};

/// Result of processing a single graph domain in parallel.
///
/// Contains the children (if any) and whether they were verified or need
/// further splitting, plus information about domains that couldn't be split
/// (no unstable neurons left).
#[derive(Debug)]
pub(super) enum GraphDomainResult {
    /// Domain was already verified (per configured verification direction).
    AlreadyVerified,
    /// Domain conclusively violates the property (per configured verification direction).
    Violation,
    /// Children created (each child has bounds and verification status)
    Children(Vec<(GraphBabDomain, bool)>), // (domain, is_verified)
    /// No unstable neurons - domain cannot be split
    NoUnstable {
        lower: f32,
        upper: f32,
        verified: bool,
    },
    /// Child propagation failed — domain is unresolved (#1861).
    /// The BaB loop MUST NOT return Verified while any domain has this status,
    /// because it means part of the input space was not explored.
    PropagationFailure,
}

/// Result of processing a batch of multi-objective domains in parallel GPU mode.
#[derive(Debug)]
pub(super) enum MultiObjectiveGraphDomainResult {
    /// Domain was already verified (all objectives verified).
    AlreadyVerified,
    /// Domain conclusively violates the property (any objective violated).
    Violation,
    /// Children created (each child has bounds and verification status)
    Children(Vec<(MultiObjectiveGraphBabDomain, bool)>), // (domain, all_verified)
    /// Children created, but at least one SIBLING was dropped as a "conclusive
    /// violation" (#violdrop). The surviving children must still be enqueued —
    /// a drop on one child says nothing about its siblings' sub-regions — while
    /// the drop is still recorded so the final verdict cannot claim `Verified`
    /// for the abandoned region.
    ///
    /// The batched GPU lane used to REPLACE the whole `Children` result with
    /// [`Self::Violation`], discarding every surviving sibling with it; the
    /// sequential lane never had that defect (its `ChildOutcome::Dropped` is
    /// per-child and the loop keeps processing the remaining children). Only
    /// reachable with the legacy drop armed (`NY_BAB_DROP_VIOLATED_CHILD=1`),
    /// which is off by default.
    ChildrenWithViolatedDrop(Vec<(MultiObjectiveGraphBabDomain, bool)>),
    /// No unstable neurons - domain cannot be split.
    /// #1866: Now carries `any_violated` so the BaB loop can distinguish
    /// violation (PotentialViolation) from unresolved (Unknown).
    NoUnstable {
        all_verified: bool,
        any_violated: bool,
    },
    /// Child propagation failed — domain is unresolved (#1861).
    PropagationFailure,
    /// At least one child of this parent was not evaluated because the
    /// authoritative BaB deadline (or its safe GPU admission reserve) refused
    /// the next cooperative chunk. This is distinct from a numerical
    /// propagation failure and must dominate every partial result for the
    /// parent; the outer verifier terminates with `Timeout`.
    DeadlineExpired,
}
