// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain list with TensorStorage-backed pick_out/add operations.
//!
//! This module provides `DomainList`, a dynamic storage structure for branch-and-bound
//! domains following alpha-beta-CROWN's `BatchedDomainList` pattern. It enables
//! efficient batch operations:
//!
//! - `pick_out(N)`: Extract N domains for GPU processing
//! - `add(results)`: Add processed domains back to storage
//! - `sort_by_domain_priority()`: Reorder domains by CPU-BaB queue priority
//!
//! # Architecture
//!
//! ```text
//! DomainList (CPU storage)
//!   ├── layer_bounds: HashMap<String, TensorStorage>  // [batch, *shape]
//!   ├── input_bounds: (TensorStorage, TensorStorage)  // lower, upper
//!   ├── global_lbs: TensorStorage                     // [batch]
//!   └── metadata: Vec<DomainMetadata>                 // histories, depths
//!           ↓ pick_out(N)
//!   PickedDomains (for GPU transfer)
//!   └── All bounds tensors with batch=N
//!           ↓ GPU processing
//!   ProcessedDomains (results from GPU)
//!           ↓ add()
//!   DomainList (updated)
//! ```
//!
//! # Module Structure
//!
//! - [`types`]: Core data types (`CachedLinearBounds`, `DomainMetadata`, `DomainListConfig`)
//! - [`picked`]: `PickedDomains` and branch selection methods
//! - [`processed`]: `ProcessedDomains` and GPU result constructors
//! - [`storage`]: `DomainList` CRUD operations (`new`, `pick_out`, `add`)
//! - [`ordering`]: `DomainList` sorting (`sort`) and in-place permutation
//! - [`filter`]: Batch filtering utility (`filter_batch`)
//!
//! # Reference
//!
//! Based on alpha-beta-CROWN: `complete_verifier/branching_domains.py`

mod alpha_queue;
mod cached_linear_bounds;
mod eviction;
mod filter;
mod grouped;
mod memory;
mod ordering;
pub mod picked;
pub mod processed;
mod storage;
pub mod types;

#[cfg(test)]
use grouped::{
    evaluate_grouped_empty_for_test, evaluate_grouped_queued_for_test,
    evaluate_grouped_roots_for_test, evaluate_grouped_verified_for_test, PackedGroupedBounds,
    TestGroupedRootDisposition,
};
pub(crate) use grouped::{
    EvaluatedGroupedChild, EvaluatedGroupedRoots, GroupedBatchCompletion, GroupedBoundSummary,
    GroupedChildEvaluationToken, GroupedDisjunctiveLayout, GroupedDomainId, GroupedParentOutcome,
    GroupedParentResolution, GroupedQueueStatus, GroupedRootEvaluationToken,
    GroupedSpecFingerprint, PickedGroupedDomains,
};
pub use picked::PickedDomains;
pub use processed::ProcessedDomains;
pub use types::{CachedLinearBounds, DomainListConfig, DomainMetadata};

// Test-only re-exports (tests/pick_add.rs and tests/permutation.rs via `use super::*`)
#[cfg(test)]
pub(crate) use filter::filter_batch;
#[cfg(test)]
pub(crate) use ordering::apply_permutation;

use ny_tensor::TensorStorage;
use std::collections::HashMap;

use types::DomainListConfig as Config;

#[derive(Debug, Clone, Copy, Default)]
struct QueueEvictionPolicy {
    /// Estimated live-frontier byte cap. Zero disables byte enforcement.
    max_queue_bytes: usize,
    /// Queue priority sense used to decide which domains survive eviction.
    verify_upper_bound: bool,
}

/// Dynamic storage for branch-and-bound domains with pick_out/add pattern.
pub struct DomainList {
    /// Configuration.
    pub(crate) config: Config,
    /// Immutable process-local identity binding packed alpha state to this graph-local queue.
    alpha_queue_identity: u64,
    /// Per-layer lower bounds storage.
    pub(crate) layer_lowers: HashMap<String, Box<dyn TensorStorage + Send>>,
    /// Per-layer upper bounds storage.
    pub(crate) layer_uppers: HashMap<String, Box<dyn TensorStorage + Send>>,
    /// Input lower bounds storage.
    pub(crate) input_lowers: Box<dyn TensorStorage + Send>,
    /// Input upper bounds storage.
    pub(crate) input_uppers: Box<dyn TensorStorage + Send>,
    /// Global lower bounds (objective) storage: shape [batch, 1].
    pub(crate) global_lbs: Box<dyn TensorStorage + Send>,
    /// Global upper bounds (objective) storage: shape [batch, 1].
    pub(crate) global_ubs: Box<dyn TensorStorage + Send>,
    /// Optional packed OR-of-AND row state. The current production executor
    /// leaves this `None`; the grouped GPU route remains default-off pending
    /// qualification.
    grouped: Option<grouped::GroupedDisjunctiveStorage>,
    /// Per-domain metadata (constraints, depths) - list-based, not tensor.
    pub(crate) metadata: Vec<DomainMetadata>,
    /// Cumulative count of unverified domains removed by queue-cap eviction.
    /// Nonzero means the search space was truncated: the BaB loop must not
    /// report Verified on queue exhaustion (see `evict_excess_domains`).
    pub(crate) evicted: usize,
    /// Private queue policy configured by the owning graph verifier.
    queue_eviction_policy: QueueEvictionPolicy,
}

#[cfg(test)]
mod tests;
