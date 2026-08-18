// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain eviction logic for bounded-queue BaB (#2326).
//!
//! Count and byte caps share one priority-preserving compaction path. Every
//! stored domain is unverified, so eviction discards unexplored search space:
//! the cumulative `evicted` count must make the BaB result Unknown rather than
//! Verified when the queue later drains.

use super::filter::filter_batch;
use super::types::DomainMetadata;
use super::DomainList;
use crate::beta_crown::config::BetaCrownConfig;
use ny_core::{NyError, Result};
use std::collections::HashSet;

impl DomainList {
    /// Configure model-aware queue eviction for the owning verification run.
    ///
    /// The public `DomainListConfig` remains count-only for compatibility.
    /// GPU BaB installs its byte cap and verification direction through this
    /// private policy before adding the root, after which every `add()` reaches
    /// `evict_excess_domains` and recomputes from the current row payload.
    pub(crate) fn configure_queue_eviction(
        &mut self,
        max_queue_bytes: usize,
        verify_upper_bound: bool,
    ) -> Result<()> {
        if self.grouped.is_some() && verify_upper_bound {
            return Err(NyError::InvalidSpec(
                "grouped disjunctive DomainList only supports lower-margin verification"
                    .to_string(),
            ));
        }
        if self.grouped.is_some() && max_queue_bytes > 0 {
            return Err(NyError::InvalidSpec(
                "max_queue_bytes is not supported for grouped disjunctive DomainList: \
                 grouped row-sidecar bytes are not part of the scalar queue census"
                    .to_string(),
            ));
        }
        self.queue_eviction_policy.max_queue_bytes = max_queue_bytes;
        self.queue_eviction_policy.verify_upper_bound = verify_upper_bound;
        self.evict_excess_domains()
    }

    /// Evict lowest-priority domains when either resident cap is exceeded.
    ///
    /// `max_queue_size` remains the configured count cap. `max_queue_bytes`
    /// scans the current row census in actual queue-priority order, retaining
    /// the longest prefix that fits. The first domain is always retained for
    /// forward progress even when it alone exceeds the byte budget.
    ///
    /// Reference: Issue #2326 Finding 1
    pub(super) fn evict_excess_domains(&mut self) -> Result<()> {
        self.validate_grouped_alignment()?;
        let len = self.metadata.len();
        if len == 0 {
            return Ok(());
        }

        let count_cap = self.config.max_queue_size;
        let byte_cap = self.queue_eviction_policy.max_queue_bytes;
        if count_cap == 0 && byte_cap == 0 {
            return Ok(());
        }

        let bytes_before = if byte_cap > 0 {
            self.estimated_resident_bytes()
        } else {
            0
        };
        let count_exceeded = count_cap > 0 && len > count_cap;
        let bytes_exceeded = byte_cap > 0 && bytes_before > byte_cap && len > 1;
        if !count_exceeded && !bytes_exceeded {
            return Ok(());
        }

        let priorities = self
            .metadata
            .iter()
            .map(|metadata| {
                BetaCrownConfig::domain_priority_for_mode(
                    self.queue_eviction_policy.verify_upper_bound,
                    metadata.lower_bound(),
                    metadata.upper_bound(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut indices: Vec<usize> = (0..len).collect();
        // Highest queue priority first; original index breaks ties
        // deterministically and the rebuild mask preserves original storage
        // order among the selected set.
        indices.sort_unstable_by(|&a, &b| {
            priorities[b]
                .total_cmp(&priorities[a])
                .then_with(|| a.cmp(&b))
        });

        let count_keep = if count_cap == 0 {
            len
        } else {
            count_cap.min(len)
        };
        let keep = if byte_cap == 0 {
            count_keep
        } else {
            let mut kept = 0usize;
            let mut retained_bytes = 0usize;
            for &index in indices.iter().take(count_keep) {
                let row_bytes = self.estimated_row_bytes(&self.metadata[index]);
                if kept > 0 && retained_bytes.saturating_add(row_bytes) > byte_cap {
                    break;
                }
                retained_bytes = retained_bytes.saturating_add(row_bytes);
                kept += 1;
            }
            kept.max(1)
        };
        let excess = len - keep;
        if excess == 0 {
            return Ok(());
        }
        let evicted_total = self.evicted.checked_add(excess).ok_or_else(|| {
            NyError::InternalError("DomainList eviction counter overflow".to_string())
        })?;

        // Indices to evict are the suffix after the retained priority prefix.
        let evict_set: HashSet<usize> = indices[keep..].iter().copied().collect();

        // Build keep mask for tensor filtering
        let keep_mask: Vec<bool> = (0..len).map(|i| !evict_set.contains(&i)).collect();

        // Rebuild metadata
        let new_metadata: Vec<DomainMetadata> = self
            .metadata
            .drain(..)
            .enumerate()
            .filter(|(i, _)| !evict_set.contains(i))
            .map(|(_, m)| m)
            .collect();
        self.metadata = new_metadata;

        // Rebuild tensor storages by popping all and re-adding kept items
        let total = keep_mask.len();
        self.rebuild_storage_with_mask(&keep_mask, total)?;
        self.validate_grouped_alignment()?;

        // Record the truncation: the evicted domains were unverified, so the
        // BaB result may no longer claim Verified on queue exhaustion.
        self.evicted = evicted_total;

        tracing::info!(
            evicted = excess,
            remaining = self.metadata.len(),
            max_queue_size = count_cap,
            max_queue_bytes = byte_cap,
            estimated_bytes_before = bytes_before,
            estimated_bytes_after = self.estimated_resident_bytes(),
            verify_upper_bound = self.queue_eviction_policy.verify_upper_bound,
            "DomainList: evicted lowest-priority domains to stay within queue cap"
        );

        Ok(())
    }

    /// Rebuild all tensor storages, keeping only items where `keep_mask[i]` is true.
    pub(super) fn rebuild_storage_with_mask(
        &mut self,
        keep_mask: &[bool],
        total: usize,
    ) -> Result<()> {
        // Pop all items from each storage, filter, and re-append
        for name in &self.config.layer_names {
            if let Some(storage) = self.layer_lowers.get_mut(name) {
                let all = storage.pop(total)?;
                let filtered = filter_batch(&all, keep_mask)?;
                storage.append(&filtered)?;
            }
            if let Some(storage) = self.layer_uppers.get_mut(name) {
                let all = storage.pop(total)?;
                let filtered = filter_batch(&all, keep_mask)?;
                storage.append(&filtered)?;
            }
        }

        let all_input_lowers = self.input_lowers.pop(total)?;
        let filtered_input_lowers = filter_batch(&all_input_lowers, keep_mask)?;
        self.input_lowers.append(&filtered_input_lowers)?;

        let all_input_uppers = self.input_uppers.pop(total)?;
        let filtered_input_uppers = filter_batch(&all_input_uppers, keep_mask)?;
        self.input_uppers.append(&filtered_input_uppers)?;

        let all_global_lbs = self.global_lbs.pop(total)?;
        let filtered_global_lbs = filter_batch(&all_global_lbs, keep_mask)?;
        self.global_lbs.append(&filtered_global_lbs)?;

        let all_global_ubs = self.global_ubs.pop(total)?;
        let filtered_global_ubs = filter_batch(&all_global_ubs, keep_mask)?;
        self.global_ubs.append(&filtered_global_ubs)?;

        if let Some(grouped) = self.grouped.as_mut() {
            let all_row_lowers = grouped.row_lowers.pop(total)?;
            let filtered_row_lowers = filter_batch(&all_row_lowers, keep_mask)?;
            grouped.row_lowers.append(&filtered_row_lowers)?;

            let all_row_uppers = grouped.row_uppers.pop(total)?;
            let filtered_row_uppers = filter_batch(&all_row_uppers, keep_mask)?;
            grouped.row_uppers.append(&filtered_row_uppers)?;
        }

        Ok(())
    }
}
