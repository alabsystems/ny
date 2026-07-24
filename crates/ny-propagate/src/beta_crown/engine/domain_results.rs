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
    /// No unstable neurons - domain cannot be split.
    /// #1866: Now carries `any_violated` so the BaB loop can distinguish
    /// violation (PotentialViolation) from unresolved (Unknown).
    NoUnstable {
        all_verified: bool,
        any_violated: bool,
    },
    /// Child propagation failed — domain is unresolved (#1861).
    PropagationFailure,
}
