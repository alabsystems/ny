// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Closed causal-decoder lowering, separate from heuristic GGUF/native loading.
//!
//! GPTNeoX and Qwen2 inference operators are lowered explicitly. Generated
//! binary32 positional tables define a pinned-table reference profile; they
//! are hashed into the manifest, not asserted to equal mathematical RoPE or a
//! deployed Torch kernel. Independent block/logit parity is a qualification
//! gate, not a numerical proof. The live graph input is embedded tokens;
//! checked token lookup binds that boundary to the original embedding table.
//!
//! The separate sealed `PretrainedDecoder`/`ReachableNextToken` path begins
//! with fixed original token IDs and certifies the whole exact-real decoder.
//! It uses analytic checked RoPE and full normalization instead of granting
//! the diagnostic mixed point graph whole-network interval authority.

mod config;
mod load;
mod lowering;
mod manifest;
mod model;
mod observation;
mod pretrained;
mod reachable;
#[cfg(test)]
mod tests;

pub use config::{DecoderArchitecture, DecoderContract, DecoderParameters};
pub use load::{load_decoder_safetensors, DecoderWeights};
pub use lowering::lower_decoder;
pub use manifest::{BlockIdentity, LoweringManifest, OperationIdentity, TensorIdentity};
pub use model::{DecoderGraph, EmbeddedInput, LoweredDecoder, PointObservation};
pub use observation::{
    DecoderBlockObservation, DecoderBlockObserver, ObservationBoundary, ObservationTensorIdentity,
    ObservedDecoderTensor,
};
pub use pretrained::{
    load_pretrained_pythia14m, PretrainedAssetIdentity, PretrainedDecoder,
    PretrainedDecoderManifest, PYTHIA14M_REVISION, PYTHIA_EXACT_REAL_PROFILE,
};
pub use reachable::{certify_reachable_next_token, ReachableDecoderReceipt, ReachableNextToken};

use ny_propagate::certify::work::{WorkBudget, WorkRefusal};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct DecoderOptions {
    pub deadline: Option<Instant>,
    pub budget: Option<WorkBudget>,
    pub max_config_bytes: usize,
    pub max_file_bytes: usize,
    pub max_resident_bytes: usize,
    pub max_scratch_bytes: usize,
    pub max_sequence: usize,
    pub max_nodes: usize,
    pub max_terms: u64,
    pub max_proposals: u64,
}
impl Default for DecoderOptions {
    fn default() -> Self {
        Self {
            deadline: None,
            budget: None,
            max_config_bytes: 1 << 20,
            max_file_bytes: 4 << 30,
            max_resident_bytes: 8 << 30,
            max_scratch_bytes: 4 << 30,
            max_sequence: 2048,
            max_nodes: 100_000,
            max_terms: 1 << 34,
            max_proposals: 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderContractRefusal {
    Config(String),
    Unsupported(String),
    Tensor(String),
    Resource(&'static str),
    Deadline,
    Io(String),
    Build(String),
    Budget(WorkRefusal),
}
impl std::fmt::Display for DecoderContractRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for DecoderContractRefusal {}
pub type Result<T> = std::result::Result<T, DecoderContractRefusal>;

pub(super) fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(DecoderContractRefusal::Resource("size overflow"))
}
pub(super) fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or(DecoderContractRefusal::Resource("size overflow"))
}
pub(super) fn checkpoint(options: &DecoderOptions) -> Result<()> {
    if options.deadline.is_some_and(|d| Instant::now() >= d) {
        Err(DecoderContractRefusal::Deadline)
    } else {
        Ok(())
    }
}
pub(super) fn budget(options: &DecoderOptions) -> Result<WorkBudget> {
    checkpoint(options)?;
    if let Some(budget) = &options.budget {
        return Ok(budget.clone());
    }
    WorkBudget::new(
        options.deadline,
        options.max_scratch_bytes,
        options.max_resident_bytes,
        options.max_terms,
        options.max_proposals,
        Duration::ZERO,
    )
    .map_err(DecoderContractRefusal::Budget)
}

pub(super) fn admit(
    options: &DecoderOptions,
    request: &ny_propagate::certify::work::WorkRequest,
) -> Result<()> {
    checkpoint(options)?;
    if request.scratch_bytes > options.max_scratch_bytes
        || request.persistent_bytes > options.max_resident_bytes
        || request.work > options.max_terms
        || request.proposals > options.max_proposals
    {
        return Err(DecoderContractRefusal::Resource("phase admission cap"));
    }
    Ok(())
}
