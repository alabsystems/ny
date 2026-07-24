// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched domain representation for GPU-accelerated branch-and-bound.
//!
//! This module provides `BatchedDomains`, a struct that packs multiple BaB domains
//! into batched tensors for efficient GPU processing. The design follows
//! alpha-beta-CROWN's `BatchedDomainList` pattern:
//!
//! - Domains are stored as batched tensors (first dimension = batch)
//! - CPU stores all domains; GPU processes working batch
//! - Enables 1000+ domain parallel processing on GPU
//! - Batched tensors use pooled CPU storage (PooledArray) to reuse buffers
//!   across batches and reduce allocation overhead
//! - Optional static intermediate bounds + unstable masks enable
//!   interm_transfer-style reuse across child domains
//!
//! Reference: alpha-beta-CROWN branching_domains.py

mod builder;
mod domain_list;
mod options;
mod picked_conversion;
mod sparse_bounds;
mod types;
mod utils;

/// Constraint tuple: (node_name, neuron_idx, is_active, split_point).
///
/// - **ReLU constraints**: `split_point = None`, `is_active = true` (x >= 0) or `false` (x <= 0)
/// - **GenBaB constraints**: `split_point = Some(pt)`, `is_active = true` (x >= pt) or `false` (x <= pt)
///
/// This unified format allows `DomainMetadata` and `BatchedDomains` to store both ReLU and GenBaB
/// constraints in a single list, preserving constraint order.
pub type ConstraintTuple = (String, usize, bool, Option<f32>);

pub use builder::BatchedDomainsBuilder;
pub use domain_list::{
    CachedLinearBounds, DomainList, DomainListConfig, DomainMetadata, PickedDomains,
    ProcessedDomains,
};
pub(crate) use domain_list::{
    EvaluatedGroupedChild, EvaluatedGroupedRoots, GroupedBatchCompletion, GroupedBoundSummary,
    GroupedChildEvaluationToken, GroupedDisjunctiveLayout, GroupedDomainId, GroupedParentOutcome,
    GroupedParentResolution, GroupedQueueStatus, GroupedRootEvaluationToken,
    GroupedSpecFingerprint, PickedGroupedDomains,
};
pub use options::BatchedDomainOptions;
#[cfg(test)]
pub use sparse_bounds::SparseIntermediateBounds;
pub use types::{BatchedDomains, DomainUpdate};

#[cfg(test)]
mod tests;
