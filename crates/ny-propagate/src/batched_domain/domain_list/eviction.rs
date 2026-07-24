// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain eviction logic for bounded-queue BaB (#2326).
//!
//! When the domain list exceeds `max_queue_size`, the domains with the
//! highest `lower_bound` (lowest queue priority in verify-lower mode) are
//! evicted. Every stored domain is unverified, so eviction discards
//! unexplored search space: the cumulative `evicted` count must make the BaB
//! result Unknown rather than Verified when the queue later drains.

use super::filter::filter_batch;
use super::types::DomainMetadata;
use super::DomainList;
use crate::cmp_utils::nan_propagating_cmp;
use ny_core::Result;
use std::collections::HashSet;

impl DomainList {
    /// Evict lowest-priority domains when the queue exceeds `max_queue_size`.
    ///
    /// Domains with the highest `lower_bound` are evicted first. Evicted
    /// domains are unverified, so each eviction is recorded in `self.evicted`;
    /// the BaB loop checks `evicted_count()` on queue exhaustion and reports
    /// Unknown instead of Verified when any domain was discarded.
    ///
    /// Reference: Issue #2326 Finding 1
    pub(super) fn evict_excess_domains(&mut self) -> Result<()> {
        self.validate_grouped_alignment()?;
        let max_size = self.config.max_queue_size;
        if max_size == 0 || self.metadata.len() <= max_size {
            return Ok(());
        }

        let excess = self.metadata.len() - max_size;

        let mut indices: Vec<usize> = (0..self.metadata.len()).collect();
        // Sort so that domains to KEEP come first (lowest lower_bound = most
        // promising for verification), and domains to EVICT come last.
        // NaN sorts last (evicted first) -- NaN domains are useless.
        indices.sort_unstable_by(|&a, &b| {
            nan_propagating_cmp(
                &self.metadata[a].lower_bound(),
                &self.metadata[b].lower_bound(),
            )
        });

        // Indices to evict are the last `excess` entries after sorting.
        // We need to remove these from metadata and all tensor storages.
        let evict_set: HashSet<usize> = indices[max_size..].iter().copied().collect();

        // Build keep mask for tensor filtering
        let keep_mask: Vec<bool> = (0..self.metadata.len())
            .map(|i| !evict_set.contains(&i))
            .collect();

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
        self.evicted += excess;

        tracing::info!(
            evicted = excess,
            remaining = self.metadata.len(),
            max_queue_size = max_size,
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
