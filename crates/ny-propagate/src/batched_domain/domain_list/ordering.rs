// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain list ordering, sorting, and permutation utilities.
//!
//! Provides `DomainList::sort_by_domain_priority` and the in-place
//! `apply_permutation` helper.

use super::DomainList;
use crate::BetaCrownConfig;
use ny_core::Result;
use ny_tensor::TreeTraversal;

impl DomainList {
    fn sort_with_scores(&mut self, sort_scores: Vec<f32>) -> Result<()> {
        self.validate_grouped_alignment()?;
        if self.len() <= 1 {
            return Ok(());
        }

        let mut indices: Vec<usize> = (0..sort_scores.len()).collect();
        match self.config.traversal {
            TreeTraversal::DepthFirst => {
                // Ascending: lowest score first, highest score last (popped first by stack).
                // Uses total_cmp for deterministic NaN handling — NaN sorts after all
                // finite values, preventing sort transitivity violations (#2246).
                indices.sort_by(|&a, &b| sort_scores[a].total_cmp(&sort_scores[b]));
            }
            TreeTraversal::BreadthFirst => {
                // Descending: highest score first (popped first by queue).
                // Uses total_cmp for deterministic NaN handling (#2246).
                indices.sort_by(|&a, &b| sort_scores[b].total_cmp(&sort_scores[a]));
            }
        }

        // Reorder all storages
        let n = indices.len();
        self.global_lbs.reorder(n, &indices)?;
        self.global_ubs.reorder(n, &indices)?;
        self.input_lowers.reorder(n, &indices)?;
        self.input_uppers.reorder(n, &indices)?;
        if let Some(grouped) = self.grouped.as_mut() {
            grouped.row_lowers.reorder(n, &indices)?;
            grouped.row_uppers.reorder(n, &indices)?;
        }

        for storage in self.layer_lowers.values_mut() {
            storage.reorder(n, &indices)?;
        }
        for storage in self.layer_uppers.values_mut() {
            storage.reorder(n, &indices)?;
        }

        // Reorder metadata in-place using cycle-based permutation.
        // Avoids cloning all DomainMetadata (which can contain large alpha_state
        // and cached_la HashMaps) by only swapping elements along permutation cycles.
        apply_permutation(&mut self.metadata, &mut indices)?;
        Ok(())
    }

    /// Sort domains by the same queue priority used by CPU BaB.
    ///
    /// Delegates to `BetaCrownConfig::domain_priority_for_mode` so the
    /// DomainList frontier uses the exact same priority contract (including
    /// NaN rejection) as the CPU heap path (#4406).
    ///
    /// Keeping the DomainList queue on the same priority contract as the
    /// CPU heap avoids GPU-BaB exploring a different frontier after each
    /// periodic re-sort (#3870).
    pub fn sort_by_domain_priority(&mut self, verify_upper_bound: bool) -> Result<()> {
        if self.grouped.is_some() && verify_upper_bound {
            return Err(ny_core::NyError::InvalidSpec(
                "grouped disjunctive DomainList only supports lower-margin verification"
                    .to_string(),
            ));
        }
        if self.metadata.len() != self.len() {
            return Err(ny_core::NyError::InternalError(format!(
                "sort_by_domain_priority: metadata length {} != domain count {}",
                self.metadata.len(),
                self.len(),
            )));
        }
        // Derive sort scores from DomainMetadata via the shared helper,
        // which rejects NaN bounds with an explicit error rather than
        // allowing silent NaN propagation through the sort (#4406).
        // Metadata stays aligned with domain storage through the same
        // keep_mask contract that add() and reorder() maintain.
        let sort_scores: Vec<f32> = self
            .metadata
            .iter()
            .map(|m| {
                BetaCrownConfig::domain_priority_for_mode(
                    verify_upper_bound,
                    m.lower_bound(),
                    m.upper_bound(),
                )
            })
            .collect::<Result<Vec<f32>>>()?;
        self.sort_with_scores(sort_scores)
    }
}

/// Apply a permutation to a Vec in-place using cycle-based swaps.
///
/// Given `perm[i] = j`, the result is that `data[i]` will contain the element
/// that was originally at `data[j]`. This matches the semantics of the previous
/// clone-based approach: `sorted[i] = original[perm[i]]`.
///
/// The `perm` slice is modified (each entry set to identity) as a visited
/// marker to avoid allocating a separate boolean array.
pub(crate) fn apply_permutation<T>(data: &mut [T], perm: &mut [usize]) -> Result<()> {
    let n = data.len();
    if perm.len() != n {
        return Err(ny_core::NyError::InternalError(format!(
            "apply_permutation: permutation length {} != data length {}",
            perm.len(),
            n,
        )));
    }
    for i in 0..n {
        // Follow the cycle starting at position i
        if perm[i] == i {
            continue;
        }
        let mut j = i;
        loop {
            let target = perm[j];
            if target == i {
                // End of cycle — mark final link as identity
                perm[j] = j;
                break;
            }
            data.swap(j, target);
            perm[j] = j; // Mark as visited
            j = target;
        }
    }
    Ok(())
}
