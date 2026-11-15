// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Complete-vocabulary affine-head certificates over a supplied hidden-state box.
//!
//! This is D2a: the caller must independently establish that the hidden box
//! encloses the trajectory. It neither lowers a decoder nor certifies omitted
//! normalization, tokenization, RoPE, logit processors or upstream arithmetic.
//! Every competitor is covered; strict negative logit differences certify the
//! target, while exact ties and unresolved competitors remain unknown.
//!
//! Preparation checks original coefficients and charges full-vocabulary work.
//! Its immutable borrowed head and checked group geometry can be reused for
//! boxes around the same anchor. Exported evidence is untrusted until replayed
//! against the original head, box, anchor and numerical semantics.

pub mod certify;
mod execution;
pub mod model_tree;
pub mod next_token;
pub mod normalized_head;
mod numeric;
pub mod resident_head;
mod shared_head;
pub mod vocab_tree;
mod wire;

pub use certify::{
    replay_next_token, replay_next_token_json, verify_dense_next_token, CertifiedNextToken,
    NextTokenOutcome, NextTokenProvenance,
};
pub use execution::Binary32Reduction;
pub use model_tree::{
    prepare_model_vocab_tree, ModelTreePreparationStats, ModelTreeQueryStats, ModelTreeReport,
    PreparedModelVocabTree,
};
pub use next_token::{verify_next_token, NextTokenReport};
pub use resident_head::{ResidentAffineHead, ResidentAffineQuery};
pub use shared_head::SharedAffineHead;
pub use vocab_tree::{prepare_vocab_tree, PreparedVocabTree};
pub use wire::{BudgetedWireExport, WireExportStats};

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Pinned affine-head arithmetic. No variant authorizes arbitrary kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeadArithmetic {
    /// Exact real affine map whose coefficients are the supplied binary64 values.
    ExactReal,
    /// Each logit starts at its bias and adds `weight[k] * h[k]` in ascending
    /// coordinate order using separate round-to-nearest binary64 operations.
    /// FMA, quantization, soft caps and logit processors are outside this contract.
    SequentialBinary64,
    /// Closed binary32 reduction contract over finite binary32 hidden inputs.
    /// Original coefficients must be exactly representable in binary32.
    Binary32 { reduction: Binary32Reduction },
}

/// Original row-major vocabulary head; borrowing immutable coefficients binds
/// a prepared tree to the exact head without copying a giant weight matrix.
#[derive(Debug, Clone, Copy)]
pub struct AffineHead<'a> {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub weights: &'a [f64],
    pub bias: &'a [f64],
    pub arithmetic: HeadArithmetic,
}

impl AffineHead<'_> {
    pub(super) fn row(&self, id: usize) -> &[f64] {
        &self.weights[id * self.hidden_size..(id + 1) * self.hidden_size]
    }
}

/// The independently supplied enclosure, with arbitrary asymmetric endpoints.
#[derive(Debug, Clone, Copy)]
pub struct HiddenBox<'a> {
    pub lo: &'a [f64],
    pub hi: &'a [f64],
}

/// Typed resource policy. All counts are checked before proportional allocations.
#[derive(Debug, Clone)]
pub struct VocabOptions {
    pub deadline: Option<Instant>,
    /// Shared ledger for preparation/replay; prepared queries retain its clone.
    pub budget: Option<crate::certify::work::WorkBudget>,
    pub max_vocab: usize,
    pub max_hidden: usize,
    /// Includes the original borrowed head, checked geometry and evidence.
    pub max_persistent_bytes: u64,
    pub max_scratch_bytes: u64,
    pub max_terms: u64,
    pub max_nodes: usize,
    pub max_proposals: u64,
    pub max_evidence_bytes: usize,
    pub leaf_size: usize,
}

impl Default for VocabOptions {
    fn default() -> Self {
        Self {
            deadline: None,
            budget: None,
            max_vocab: 1_000_000,
            max_hidden: 65_536,
            max_persistent_bytes: 2 << 30,
            max_scratch_bytes: 32 << 20,
            max_terms: 1 << 34,
            max_nodes: 2_000_000,
            max_proposals: 1_000_000,
            max_evidence_bytes: 64 << 20,
            leaf_size: 32,
        }
    }
}

/// Fail-closed input, numerical, resource or replay refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderRefusal {
    Shape(&'static str),
    NonFinite(&'static str),
    Resource(&'static str),
    Deadline,
    Budget(crate::certify::work::WorkRefusal),
    Evidence(&'static str),
}
impl std::fmt::Display for DecoderRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for DecoderRefusal {}

/// Original-coefficient and coverage-ID visits for cold preparation/warm
/// queries. Lease reservations are conservative predictions, not completed
/// work; these observables include complete coverage/export work. No timing
/// claim or excluded preprocessing is implied.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VocabStats {
    pub preprocessing_terms: u64,
    pub query_terms: u64,
    pub groups_visited: usize,
    pub groups_excluded: usize,
    pub dense_rows: usize,
    pub persistent_bytes: u64,
    pub scratch_bytes: u64,
}

/// Independently replayable full-cover group or original-row exclusion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupEvidence {
    pub members: Vec<usize>,
    pub anchor_token: usize,
    /// None means direct original-row box support of every member.
    pub group_bounds: Option<GroupBounds>,
    pub upper: f64,
}

/// Every candidate is independently checked before taking their minimum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupBounds {
    pub at_anchor_upper: f64,
    pub anchor_upper: f64,
    pub radius_l1: f64,
    pub radius_l2: f64,
    pub radius_linf: f64,
    pub residual_componentwise: f64,
    pub residual_l1_linf: f64,
    pub residual_l2_l2: f64,
    pub residual_linf_l1: f64,
    pub execution_error: f64,
    pub upper: f64,
}

/// Plain serialized evidence conveys no certification authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VocabEvidence {
    pub schema_version: u32,
    pub scope_digest: [u8; 32],
    pub target: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub groups: Vec<GroupEvidence>,
}

impl VocabEvidence {
    /// Default wire export uses schema v2 maximal original-token-ID ranges.
    /// Internal evidence and direct serde serialization remain schema v1.
    /// Raw serialization has no WorkBudget/deadline; use export_json for
    /// resource-admitted production export. Bytes carry no proof authority.
    pub fn to_json(&self) -> Result<String, DecoderRefusal> {
        serde_json::to_string(&wire::CompactEvidence(self))
            .map_err(|_| DecoderRefusal::Evidence("serialization"))
    }

    /// Compatibility export for consumers of the explicit-ID v1 wire format.
    pub fn to_json_v1(&self) -> Result<String, DecoderRefusal> {
        serde_json::to_string(self).map_err(|_| DecoderRefusal::Evidence("serialization"))
    }

    /// Raw stream export without a WorkBudget/deadline or copied memberships.
    /// The supplied writer controls output capacity; replay separately admits
    /// the complete original membership and all arithmetic work.
    pub fn write_json(&self, writer: impl std::io::Write) -> Result<(), DecoderRefusal> {
        wire::write_compact_json(self, writer)
    }

    /// Admit output memory/work on the existing shared Replay ledger, poll
    /// bounded member chunks, and retain the byte lease across replay. Actual
    /// serialization visits/ranges are separate from query arithmetic stats.
    pub fn export_json(
        &self,
        options: &VocabOptions,
    ) -> Result<BudgetedWireExport, DecoderRefusal> {
        wire::export_with_budget(self, options)
    }
}

pub(super) struct Meter<'a> {
    opts: &'a VocabOptions,
    terms: u64,
    budget: Option<crate::certify::work::WorkBudget>,
    class: crate::certify::work::LeaseClass,
}
impl<'a> Meter<'a> {
    pub(super) fn new(opts: &'a VocabOptions) -> Result<Self, DecoderRefusal> {
        if opts.leaf_size == 0 || opts.max_nodes == 0 {
            return Err(DecoderRefusal::Resource("zero leaf or node limit"));
        }
        let out = Self {
            opts,
            terms: 0,
            budget: None,
            class: crate::certify::work::LeaseClass::Optional,
        };
        out.checkpoint()?;
        Ok(out)
    }
    pub(super) fn with_budget(
        opts: &'a VocabOptions,
        budget: crate::certify::work::WorkBudget,
    ) -> Result<Self, DecoderRefusal> {
        Self::with_budget_class(opts, budget, crate::certify::work::LeaseClass::Optional)
    }
    pub(super) fn with_budget_class(
        opts: &'a VocabOptions,
        budget: crate::certify::work::WorkBudget,
        class: crate::certify::work::LeaseClass,
    ) -> Result<Self, DecoderRefusal> {
        let mut out = Self::new(opts)?;
        out.budget = Some(budget);
        out.class = class;
        out.checkpoint()?;
        Ok(out)
    }
    pub(super) fn checkpoint(&self) -> Result<(), DecoderRefusal> {
        if let Some(budget) = &self.budget {
            budget
                .checkpoint_for(self.class)
                .map_err(DecoderRefusal::Budget)?;
        }
        if self.opts.deadline.is_some_and(|d| Instant::now() >= d) {
            Err(DecoderRefusal::Deadline)
        } else {
            Ok(())
        }
    }
    pub(super) fn charge(&mut self, terms: usize) -> Result<(), DecoderRefusal> {
        self.checkpoint()?;
        self.terms = self
            .terms
            .checked_add(terms as u64)
            .ok_or(DecoderRefusal::Resource("work overflow"))?;
        if self.terms > self.opts.max_terms {
            Err(DecoderRefusal::Resource("work cap"))
        } else {
            Ok(())
        }
    }
    pub(super) fn terms(&self) -> u64 {
        self.terms
    }
}

#[cfg(test)]
mod provenance_tests;
#[cfg(test)]
mod shared_head_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod wire_tests;
