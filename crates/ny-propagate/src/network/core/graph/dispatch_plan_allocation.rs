// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Complete owned-payload capacity observation for ONE existing dispatch plan.
//! This borrows the original plan; it neither builds a plan nor authorizes a
//! finalized graph/root, query, provider or finite traversal. Graph/layer/cache
//! and domain-map ownership remains a separate complete-capture obligation.

use super::super::structural_map::TrackedStringMapAllocationFactV1;
use super::*;
use crate::amortise::CooperativeWork;
use std::mem::size_of;

#[must_use]
pub(crate) struct DispatchPlanAllocationReceiptV2<'a> {
    source: &'a CrownDispatchPlan,
    name_index: TrackedStringMapAllocationFactV1,
    index_key_capacity_bytes: usize,
    name_index_owner_allocation_bytes: usize,
    vector_capacity_bytes: usize,
    index_name_capacity_bytes: usize,
    heap_charge_bytes: usize,
}

impl DispatchPlanAllocationReceiptV2<'_> {
    pub(crate) fn matches_source(&self, source: &CrownDispatchPlan) -> bool {
        std::ptr::eq(self.source, source)
    }
    pub(crate) fn source_inline_bytes(&self) -> usize {
        size_of::<CrownDispatchPlan>()
    }
    pub(crate) fn name_index_owner_identity(&self) -> usize {
        self.source.name_to_idx.owner_identity()
    }
    pub(crate) fn name_index_owner_allocation_bytes(&self) -> usize {
        self.name_index_owner_allocation_bytes
    }
    pub(crate) fn name_index_table(&self) -> &TrackedStringMapAllocationFactV1 {
        &self.name_index
    }
    pub(crate) fn index_key_capacity_bytes(&self) -> usize {
        self.index_key_capacity_bytes
    }
    pub(crate) fn vector_capacity_bytes(&self) -> usize {
        self.vector_capacity_bytes
    }
    pub(crate) fn index_name_capacity_bytes(&self) -> usize {
        self.index_name_capacity_bytes
    }
    /// Four Vec allocations, every separately owned String key/name capacity,
    /// the shared Arc/map owner, and the tracked table high-water envelope, excluding the source inline
    /// object (already contained in a GraphNetwork's cached OnceLock slot).
    pub(crate) fn heap_charge_bytes_excluding_inline(&self) -> usize {
        self.heap_charge_bytes
    }
}

fn overflow() -> NyError {
    NyError::InvalidSpec(
        "dispatch-plan capacity sum overflow or unknown String-key capacity".into(),
    )
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(overflow)
}
fn vector<T>(capacity: usize) -> Result<usize> {
    capacity.checked_mul(size_of::<T>()).ok_or_else(overflow)
}

impl CrownDispatchPlan {
    /// Shared name-index owners may be conservatively counted by every plan
    /// receipt; use their identity to deduplicate in a complete containing census.
    /// Quote the existing immutable artifact, with original caller work before
    /// every metadata batch. Fixed-cost map facts avoid scanning empty buckets
    /// of a sparse table; no public HashMap capacity estimate or payload copy.
    /// Each String's inline header is included by its owning table/Vec, while
    /// its separate capacity is added exactly once. Errors may allocate.
    pub(crate) fn allocation_receipt_v2<P: CooperativeWork>(
        &self,
        work: &mut P,
    ) -> Result<DispatchPlanAllocationReceiptV2<'_>> {
        work.checkpoint()?;
        work.before(64)?;
        let name_index_owner_allocation_bytes = self
            .name_to_idx
            .owner_allocation_request_bytes()
            .ok_or_else(overflow)?;
        let name_index = self.name_to_idx.allocation_fact_v1();
        let index_key_capacity_bytes = self
            .name_to_idx
            .key_payload_capacity_bytes_v2()
            .ok_or_else(overflow)?;
        let vector_capacity_bytes = add(
            add(
                vector::<usize>(self.exec_order.capacity())?,
                vector::<usize>(self.reverse_order.capacity())?,
            )?,
            add(
                vector::<String>(self.idx_to_name.capacity())?,
                vector::<DispatchRoute>(self.routes.capacity())?,
            )?,
        )?;
        let mut index_name_capacity_bytes = 0;
        for name in &self.idx_to_name {
            work.before(8)?;
            index_name_capacity_bytes = add(index_name_capacity_bytes, name.capacity())?;
        }
        let heap_charge_bytes = add(
            add(
                name_index.conservative_table_high_water_charge_bytes(),
                add(index_key_capacity_bytes, name_index_owner_allocation_bytes)?,
            )?,
            add(vector_capacity_bytes, index_name_capacity_bytes)?,
        )?;
        work.checkpoint()?;
        Ok(DispatchPlanAllocationReceiptV2 {
            source: self,
            name_index,
            index_key_capacity_bytes,
            name_index_owner_allocation_bytes,
            vector_capacity_bytes,
            index_name_capacity_bytes,
            heap_charge_bytes,
        })
    }
}

#[cfg(test)]
#[path = "dispatch_plan_allocation_tests.rs"]
mod tests;
