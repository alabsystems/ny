// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain processing and split handling.

mod child;
mod clip;
mod processing;
mod strengthening;

use crate::beta_crown::domain::BabDomain;

/// Result of processing a single domain, carrying both children and failure info.
/// #1865: Prevents silent domain dropping by explicitly tracking propagation failures
/// and no-branch conditions that would otherwise cause false Verified results.
pub(crate) struct DomainProcessingResult {
    pub children: Vec<BabDomain>,
    /// A child creation call returned Err — input sub-region unexplored.
    pub had_propagation_failure: bool,
    /// No unstable neurons to branch on — domain is unresolved.
    pub had_no_branch: bool,
    /// No input dimension admits a split — domain is unresolved.
    pub had_unsplittable: bool,
}

#[cfg(test)]
mod tests;
