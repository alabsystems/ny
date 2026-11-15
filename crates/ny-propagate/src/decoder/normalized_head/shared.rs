// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Same-original-owner normalized queries. Only coefficient residency is shared;
//! every original hash/row arithmetic scan is charged again for independent replay.
use super::*;

/// Check a supplied pre-normalization box under the original head's session.
/// Limits cannot replace that session's work, deadline or resident authority.
pub fn verify_shared_normalized_next_token(
    owner: &SharedAffineHead,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    limits: &VocabOptions,
) -> Result<NormalizedHeadReport> {
    let ledger = owner.budget();
    run(
        owner.as_head(),
        norm,
        pre,
        target,
        None,
        NormalizedHeadOptions {
            budget: &ledger,
            limits,
        },
        LeaseClass::Optional,
        NextTokenProvenance::CheckedQuery,
        Some(owner.clone()),
    )
}

/// Complete original-coefficient reconstruction on the owner's existing ledger.
pub fn replay_shared_normalized_next_token(
    owner: &SharedAffineHead,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    evidence: &NormalizedHeadEvidence,
    limits: &VocabOptions,
) -> Result<NormalizedHeadReport> {
    let ledger = owner.budget();
    run(
        owner.as_head(),
        norm,
        pre,
        target,
        Some(evidence),
        NormalizedHeadOptions {
            budget: &ledger,
            limits,
        },
        LeaseClass::Replay,
        NextTokenProvenance::OriginalReplay,
        Some(owner.clone()),
    )
}

#[cfg(test)]
mod tests;
