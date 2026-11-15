// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Explicit original-head residency with caller-admitted finite query work.
//!
//! This is an additional capability: ordinary SharedAffineHead sessions keep
//! their original cumulative work/deadline ledger. No query method creates or
//! replaces a budget. Resident and query owners join one configured byte
//! authority, which counts participating capacities rather than process RSS.

use super::{
    BudgetedWireExport, DecoderRefusal, HeadArithmetic, HiddenBox, NextTokenReport,
    SharedAffineHead, VocabEvidence, VocabOptions,
};
use crate::certify::residency::MemoryAuthority;
use crate::certify::work::WorkBudget;

/// Owns immutable original coefficients and their original resident claim.
/// Construction moves vectors once and validates/hashes every original value.
#[derive(Clone, Debug)]
pub struct ResidentAffineHead {
    original: SharedAffineHead,
    memory: MemoryAuthority,
}

impl ResidentAffineHead {
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        weights: Vec<f64>,
        bias: Vec<f64>,
        arithmetic: HeadArithmetic,
        construction: &VocabOptions,
    ) -> Result<Self, DecoderRefusal> {
        let ledger = construction
            .budget
            .as_ref()
            .ok_or(DecoderRefusal::Resource(
                "resident construction requires an explicit caller ledger",
            ))?;
        let memory = ledger.memory_authority().ok_or(DecoderRefusal::Resource(
            "resident construction requires a shared memory authority",
        ))?;
        let original = SharedAffineHead::new(
            vocab_size,
            hidden_size,
            weights,
            bias,
            arithmetic,
            construction,
        )?;
        Ok(Self { original, memory })
    }

    pub fn resident_bytes(&self) -> usize {
        self.original.reserved_bytes()
    }

    pub fn memory_authority(&self) -> MemoryAuthority {
        self.memory.clone()
    }

    /// Bind one already admitted finite query. Other phases and replay must
    /// use this exact work ledger, not another equal-limit ledger. The returned
    /// context keeps original coefficients and its metadata reservation alive.
    pub fn bind_query(&self, query: &WorkBudget) -> Result<ResidentAffineQuery, DecoderRefusal> {
        let requested = query.memory_authority().ok_or(DecoderRefusal::Resource(
            "resident query requires an explicit shared memory authority",
        ))?;
        if !self.memory.shares_authority(&requested) {
            return Err(DecoderRefusal::Resource(
                "resident query memory authority mismatch",
            ));
        }
        Ok(ResidentAffineQuery {
            owner: self.original.bind_resident_query(query)?,
        })
    }
}

/// Query-bound private owner context. It exposes no unrestricted Shared head,
/// parameter-vector materializer, or method to renew a spent query budget.
#[derive(Clone, Debug)]
pub struct ResidentAffineQuery {
    owner: SharedAffineHead,
}

impl ResidentAffineQuery {
    pub fn budget(&self) -> WorkBudget {
        self.owner.budget()
    }

    pub fn resident_head_bytes(&self) -> usize {
        self.owner.reserved_bytes()
    }

    pub fn verify_dense_next_token(
        &self,
        hidden: HiddenBox<'_>,
        anchor: &[f64],
        target: usize,
        options: &VocabOptions,
    ) -> Result<NextTokenReport, DecoderRefusal> {
        self.owner
            .verify_dense_next_token(hidden, anchor, target, options)
    }

    pub fn replay_next_token(
        &self,
        hidden: HiddenBox<'_>,
        anchor: &[f64],
        target: usize,
        evidence: &VocabEvidence,
        options: &VocabOptions,
    ) -> Result<NextTokenReport, DecoderRefusal> {
        self.owner
            .replay_next_token(hidden, anchor, target, evidence, options)
    }

    pub fn replay_next_token_json(
        &self,
        hidden: HiddenBox<'_>,
        anchor: &[f64],
        target: usize,
        json: &[u8],
        options: &VocabOptions,
    ) -> Result<NextTokenReport, DecoderRefusal> {
        self.owner
            .replay_next_token_json(hidden, anchor, target, json, options)
    }

    /// Conditional mathematical-real normalization and complete original head.
    /// The caller independently supplies the pre-normalization enclosure;
    /// deployed normalization and upstream trajectory remain outside this claim.
    pub fn verify_normalized_next_token(
        &self,
        norm: super::normalized_head::FinalNormalization<'_>,
        pre: HiddenBox<'_>,
        target: usize,
        options: &VocabOptions,
    ) -> Result<
        super::normalized_head::NormalizedHeadReport,
        super::normalized_head::NormalizedHeadRefusal,
    > {
        super::normalized_head::shared::verify_shared_normalized_next_token(
            &self.owner,
            norm,
            pre,
            target,
            options,
        )
    }

    pub fn replay_normalized_next_token(
        &self,
        norm: super::normalized_head::FinalNormalization<'_>,
        pre: HiddenBox<'_>,
        target: usize,
        evidence: &super::normalized_head::NormalizedHeadEvidence,
        options: &VocabOptions,
    ) -> Result<
        super::normalized_head::NormalizedHeadReport,
        super::normalized_head::NormalizedHeadRefusal,
    > {
        super::normalized_head::shared::replay_shared_normalized_next_token(
            &self.owner,
            norm,
            pre,
            target,
            evidence,
            options,
        )
    }

    pub fn replay_normalized_json(
        &self,
        norm: super::normalized_head::FinalNormalization<'_>,
        pre: HiddenBox<'_>,
        target: usize,
        json: &[u8],
        options: &VocabOptions,
    ) -> Result<
        super::normalized_head::wire::NormalizedWireReplay,
        super::normalized_head::NormalizedHeadRefusal,
    > {
        super::normalized_head::wire::replay_shared_normalized_json(
            &self.owner,
            norm,
            pre,
            target,
            json,
            options,
        )
    }

    pub fn export_normalized_json(
        &self,
        evidence: &super::normalized_head::NormalizedHeadEvidence,
        options: &VocabOptions,
    ) -> Result<
        super::normalized_head::wire::NormalizedWireExport,
        super::normalized_head::NormalizedHeadRefusal,
    > {
        super::normalized_head::wire::export_shared_normalized_json(&self.owner, evidence, options)
    }

    pub fn export_json(
        &self,
        evidence: &VocabEvidence,
        options: &VocabOptions,
    ) -> Result<BudgetedWireExport, DecoderRefusal> {
        self.owner.export_json(evidence, options)
    }
}

#[cfg(test)]
mod tests;
