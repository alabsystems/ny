// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-depth ReLU splitting configuration methods for `BetaCrownConfig`.
//!
//! Separated from `beta_config.rs` to keep that file within size limits.
//! Part of #2767.

use super::beta_config::AUTO_ENLARGE_BATCH_CAP;
use super::BetaCrownConfig;

impl BetaCrownConfig {
    /// Check whether the batch size should be enlarged after a BaB iteration.
    ///
    /// Returns `Some(new_batch_size)` when the batch should be doubled, `None` otherwise.
    /// The batch is doubled when `auto_enlarge_batch_size` is enabled, the previous
    /// batch was fully utilized (`actual_batch_size >= current_batch_size`), and the
    /// current batch size is below `AUTO_ENLARGE_BATCH_CAP`.
    ///
    /// Reference: alpha-beta-CROWN `auto_LiRPA/utils.py:348-381` (`AutoBatchSize`).
    #[inline]
    pub fn maybe_enlarge_batch_size(
        &self,
        current_batch_size: usize,
        actual_batch_size: usize,
    ) -> Option<usize> {
        if self.auto_enlarge_batch_size
            && actual_batch_size >= current_batch_size
            && current_batch_size < AUTO_ENLARGE_BATCH_CAP
        {
            Some((current_batch_size * 2).min(AUTO_ENLARGE_BATCH_CAP))
        } else {
            None
        }
    }

    /// Try to enlarge the batch size in-place after a BaB iteration.
    ///
    /// If `auto_enlarge_batch_size` is enabled and the previous batch was fully
    /// utilized, doubles `batch_size` (capped at `AUTO_ENLARGE_BATCH_CAP`) and
    /// logs the change. Returns `true` when the batch was enlarged.
    ///
    /// This is the legacy closeout helper retained for BaB routes that have
    /// not selected the independently-gated memory-aware controller. Graph
    /// ReLU-split and DomainList input-split bypass it only when
    /// `NY_ADAPTIVE_MICROBATCH_CONTROLLER=1` is also set exactly.
    pub fn try_enlarge_batch_size(
        &self,
        batch_size: &mut usize,
        actual_batch_size: usize,
        context: &str,
    ) -> bool {
        if let Some(new_size) = self.maybe_enlarge_batch_size(*batch_size, actual_batch_size) {
            tracing::debug!(
                "{context}: auto-enlarge batch_size: {old} -> {new_size}",
                old = *batch_size,
            );
            *batch_size = new_size;
            true
        } else {
            false
        }
    }

    /// Compute effective split depth based on queue size and batch size.
    ///
    /// When the BaB queue is smaller than `batch_size * min_batch_fill_ratio`,
    /// increase depth so that `queue_size * 2^depth >= min_batch`. This keeps
    /// GPU batches full by creating more child domains per iteration.
    ///
    /// Returns 1 when multi-depth is disabled (`max_relu_split_depth <= 1`)
    /// or when the queue is already large enough.
    ///
    /// Reference: alpha-beta-CROWN `get_split_depth()` in `bab.py:40-48`.
    pub fn effective_relu_split_depth(&self, queue_size: usize) -> usize {
        let min_batch = (self.batch_size as f32 * self.min_batch_fill_ratio) as usize;
        if queue_size >= min_batch || self.max_relu_split_depth <= 1 {
            return 1;
        }
        // depth = ceil(log2(min_batch / queue_size))
        let ratio = (min_batch as f32) / (queue_size.max(1) as f32);
        let depth = ratio.log2().ceil() as usize;
        depth.clamp(1, self.max_relu_split_depth)
    }
}
