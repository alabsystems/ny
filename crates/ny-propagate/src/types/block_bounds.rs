// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Block-level bounds information for transformer verification.

use super::node_bounds::NodeBoundsInfo;
use crate::bounds::nan_propagating_min;
use serde::{Deserialize, Serialize};

/// Information about bounds for a single transformer block.
///
/// Used for block-wise verification to analyze each block independently
/// without bound explosion from propagation through the entire network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockBoundsInfo {
    /// Block index (0-based).
    pub block_index: usize,
    /// Block name prefix (e.g., "layer0").
    pub block_name: String,
    /// Per-node information for this block.
    pub nodes: Vec<NodeBoundsInfo>,
    /// Input bound width at start of block (after reset).
    pub input_width: f32,
    /// Output bound width at end of block.
    pub output_width: f32,
    /// Overall sensitivity = output_width / input_width.
    pub sensitivity: f32,
    /// Q@K^T attention bound width (if zonotope applied).
    pub qk_matmul_width: Option<f32>,
    /// SwiGLU FFN bound width (if zonotope applied).
    pub swiglu_width: Option<f32>,
    /// Whether this block saturated or had NaN/inf.
    pub degraded: bool,
}

impl BlockBoundsInfo {
    /// Get status string for this block.
    pub fn status(&self) -> &'static str {
        if self.degraded {
            "DEGRADED"
        } else if self.sensitivity > 1e6 {
            "HIGH"
        } else if self.sensitivity > 1e3 {
            "MODERATE"
        } else {
            "OK"
        }
    }
}

/// Result of block-wise verification.
///
/// Each transformer block is verified independently with fresh bounds reset
/// at each block boundary. This prevents bound explosion from propagating
/// through the entire network and gives meaningful per-block sensitivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockWiseResult {
    /// Per-block information.
    pub blocks: Vec<BlockBoundsInfo>,
    /// Epsilon used for each block's input perturbation.
    pub block_epsilon: f32,
    /// Total number of blocks.
    pub total_blocks: usize,
    /// Maximum sensitivity across all blocks.
    pub max_sensitivity: f32,
    /// Number of blocks that degraded.
    pub degraded_blocks: usize,
}

impl BlockWiseResult {
    /// Generate a summary table of block-wise verification results.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Block-wise Verification (zonotope reset per block)".to_string());
        lines.push("=================================================".to_string());
        lines.push(format!(
            "{:<15} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | Status",
            "Block", "In Width", "Out Width", "Sens.", "Q@K^T", "SwiGLU"
        ));
        lines.push(format!(
            "{:-<15}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+--------",
            "", "", "", "", "", ""
        ));

        for block in &self.blocks {
            let qk_str = match block.qk_matmul_width {
                Some(w) => format!("{:.3e}", w),
                None => "-".to_string(),
            };
            let swiglu_str = match block.swiglu_width {
                Some(w) => format!("{:.3e}", w),
                None => "-".to_string(),
            };
            let marker = if block.degraded { " <<<" } else { "" };
            lines.push(format!(
                "{:<15} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>10} | {:>10} | {}{}",
                block.block_name,
                block.input_width,
                block.output_width,
                block.sensitivity,
                qk_str,
                swiglu_str,
                block.status(),
                marker
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Block epsilon: {:.2e} | Max sensitivity: {:.3e}",
            self.block_epsilon, self.max_sensitivity
        ));
        lines.push(format!(
            "Total blocks: {} | Degraded: {}",
            self.total_blocks, self.degraded_blocks
        ));

        lines.join("\n")
    }

    /// Minimum sensitivity across all blocks.
    pub fn min_sensitivity(&self) -> f32 {
        self.blocks
            .iter()
            .map(|b| b.sensitivity)
            .fold(f32::INFINITY, nan_propagating_min) // #2577: NaN sensitivity must propagate
    }

    /// Median sensitivity across all blocks.
    ///
    /// NaN values sort last via `crate::cmp_utils::nan_propagating_cmp`. If NaN
    /// reaches either median slot after sorting, the returned median is NaN
    /// (#2601).
    pub fn median_sensitivity(&self) -> f32 {
        if self.blocks.is_empty() {
            return f32::NAN;
        }
        let mut sensitivities: Vec<f32> = self.blocks.iter().map(|b| b.sensitivity).collect();
        sensitivities.sort_by(crate::cmp_utils::nan_propagating_cmp);
        let mid = sensitivities.len() / 2;
        if sensitivities.len().is_multiple_of(2) {
            // `(a+b)/2` kept verbatim: `f32::midpoint` differs once |sensitivity| >
            // f32::MAX/2 (sum overflow), and the median steers block refinement.
            #[allow(clippy::manual_midpoint)]
            {
                (sensitivities[mid - 1] + sensitivities[mid]) / 2.0
            }
        } else {
            sensitivities[mid]
        }
    }

    /// Get the worst (highest sensitivity) k blocks, sorted descending.
    /// Returns (block_index, block_name, sensitivity, output_width).
    ///
    /// NaN-sensitivity blocks sort last via `crate::cmp_utils::nan_last_descending_cmp`,
    /// so they never displace finite-valued blocks from the "worst" ranking (#2601).
    pub fn worst_k_blocks(&self, k: usize) -> Vec<(usize, String, f32, f32)> {
        let mut indexed: Vec<_> = self
            .blocks
            .iter()
            .map(|b| {
                (
                    b.block_index,
                    b.block_name.clone(),
                    b.sensitivity,
                    b.output_width,
                )
            })
            .collect();
        // Sort by sensitivity descending (highest first, NaN last — #2995)
        indexed.sort_by(|a, b| crate::cmp_utils::nan_last_descending_cmp(&a.2, &b.2));
        indexed.truncate(k);
        indexed
    }

    /// Sensitivity range (max / min).
    pub fn sensitivity_range(&self) -> f32 {
        let min = self.min_sensitivity();
        if min <= 0.0 || !min.is_finite() {
            return f32::INFINITY;
        }
        self.max_sensitivity / min
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block_info(sensitivity: f32, degraded: bool) -> BlockBoundsInfo {
        BlockBoundsInfo {
            block_index: 0,
            block_name: "block0".to_string(),
            nodes: vec![],
            input_width: 0.1,
            output_width: sensitivity * 0.1,
            sensitivity,
            qk_matmul_width: None,
            swiglu_width: None,
            degraded,
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_bounds_info_status_degraded() {
        let block = make_block_info(100.0, true);
        assert_eq!(block.status(), "DEGRADED");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_bounds_info_status_high() {
        let block = make_block_info(1e7, false);
        assert_eq!(block.status(), "HIGH");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_bounds_info_status_moderate() {
        let block = make_block_info(1e4, false);
        assert_eq!(block.status(), "MODERATE");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_bounds_info_status_ok() {
        let block = make_block_info(100.0, false);
        assert_eq!(block.status(), "OK");
    }

    fn make_block_wise_result(sensitivities: &[f32]) -> BlockWiseResult {
        let blocks: Vec<BlockBoundsInfo> = sensitivities
            .iter()
            .enumerate()
            .map(|(i, &s)| BlockBoundsInfo {
                block_index: i,
                block_name: format!("block{}", i),
                nodes: vec![],
                input_width: 0.1,
                output_width: s * 0.1,
                sensitivity: s,
                qk_matmul_width: None,
                swiglu_width: None,
                degraded: false,
            })
            .collect();
        let max_sensitivity = sensitivities
            .iter()
            .cloned()
            .fold(0.0f32, crate::bounds::nan_propagating_max);
        BlockWiseResult {
            blocks,
            block_epsilon: 0.01,
            total_blocks: sensitivities.len(),
            max_sensitivity,
            degraded_blocks: 0,
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_min_sensitivity() {
        let result = make_block_wise_result(&[10.0, 5.0, 20.0, 3.0]);
        assert!((result.min_sensitivity() - 3.0).abs() < 1e-6);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_min_sensitivity_empty() {
        let result = make_block_wise_result(&[]);
        assert!(result.min_sensitivity().is_infinite());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_median_sensitivity_odd() {
        let result = make_block_wise_result(&[10.0, 5.0, 20.0]);
        // Sorted: [5.0, 10.0, 20.0], median = 10.0
        assert!((result.median_sensitivity() - 10.0).abs() < 1e-6);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_median_sensitivity_even() {
        let result = make_block_wise_result(&[10.0, 5.0, 20.0, 15.0]);
        // Sorted: [5.0, 10.0, 15.0, 20.0], median = (10.0 + 15.0) / 2 = 12.5
        assert!((result.median_sensitivity() - 12.5).abs() < 1e-6);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_median_sensitivity_empty() {
        let result = make_block_wise_result(&[]);
        assert!(result.median_sensitivity().is_nan());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_worst_k_blocks() {
        let result = make_block_wise_result(&[10.0, 50.0, 20.0, 100.0, 5.0]);
        let worst = result.worst_k_blocks(3);
        assert_eq!(worst.len(), 3);
        // Should be sorted by sensitivity descending
        assert_eq!(worst[0].0, 3); // block3, sensitivity 100.0
        assert_eq!(worst[1].0, 1); // block1, sensitivity 50.0
        assert_eq!(worst[2].0, 2); // block2, sensitivity 20.0
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_worst_k_blocks_larger_than_available() {
        let result = make_block_wise_result(&[10.0, 5.0]);
        let worst = result.worst_k_blocks(10);
        assert_eq!(worst.len(), 2);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_sensitivity_range() {
        let result = make_block_wise_result(&[10.0, 5.0, 20.0]);
        // max=20, min=5, range=4
        assert!((result.sensitivity_range() - 4.0).abs() < 1e-6);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_sensitivity_range_zero_min() {
        // Edge case: min sensitivity is 0
        let mut result = make_block_wise_result(&[0.0, 5.0, 20.0]);
        result.max_sensitivity = 20.0;
        assert!(result.sensitivity_range().is_infinite());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_summary() {
        let result = make_block_wise_result(&[10.0, 5.0]);
        let summary = result.summary();
        assert!(summary.contains("Block-wise Verification"));
        assert!(summary.contains("block0"));
        assert!(summary.contains("block1"));
        assert!(summary.contains("Total blocks: 2"));
    }

    /// Regression: NaN sensitivity must propagate through max_sensitivity,
    /// not be silently absorbed by `>` comparison (IEEE 754: NaN > x == false).
    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_nan_sensitivity_propagates() {
        let result = make_block_wise_result(&[10.0, f32::NAN, 5.0]);
        assert!(
            result.max_sensitivity.is_nan(),
            "NaN sensitivity must propagate through max_sensitivity: got {}",
            result.max_sensitivity
        );
    }

    /// Regression: NaN sensitivity must also propagate through min_sensitivity.
    #[ntest::timeout(5000)]
    #[test]
    fn test_block_wise_result_nan_min_sensitivity_propagates() {
        let result = make_block_wise_result(&[10.0, f32::NAN, 5.0]);
        assert!(
            result.min_sensitivity().is_nan(),
            "NaN sensitivity must propagate through min_sensitivity: got {}",
            result.min_sensitivity()
        );
    }

    /// Regression test for #2601: median_sensitivity with NaN values produces
    /// deterministic ordering — NaN sorts last so the median is computed from
    /// finite values when fewer than half are NaN.
    #[ntest::timeout(5000)]
    #[test]
    fn test_median_sensitivity_nan_sorts_last_2601() {
        // [5.0, NaN, 20.0, 10.0] → sorted: [5.0, 10.0, 20.0, NaN]
        // Even count of finite-positioned: median = (10.0 + 20.0) / 2 = 15.0
        let result = make_block_wise_result(&[5.0, f32::NAN, 20.0, 10.0]);
        assert!(
            (result.median_sensitivity() - 15.0).abs() < 1e-6,
            "median with 1 NaN in 4 blocks should be 15.0, got {}",
            result.median_sensitivity()
        );
    }

    /// Regression test for #2601: when the majority of blocks have NaN
    /// sensitivity, the median itself is NaN (NaN reaches the median position).
    #[ntest::timeout(5000)]
    #[test]
    fn test_median_sensitivity_majority_nan_propagates_2601() {
        // [NaN, NaN, 5.0] → sorted: [5.0, NaN, NaN] → median at index 1 = NaN
        let result = make_block_wise_result(&[f32::NAN, f32::NAN, 5.0]);
        assert!(
            result.median_sensitivity().is_nan(),
            "median with majority NaN blocks should be NaN, got {}",
            result.median_sensitivity()
        );
    }

    /// Regression test for #2601: with an even number of blocks, NaN in either
    /// median slot propagates into the averaged median.
    #[ntest::timeout(5000)]
    #[test]
    fn test_median_sensitivity_half_nan_propagates_2601() {
        // [1.0, 2.0, NaN, NaN] stays ordered after NaN-last sort, so the upper
        // median slot is NaN and the averaged median is NaN.
        let result = make_block_wise_result(&[1.0, 2.0, f32::NAN, f32::NAN]);
        assert!(
            result.median_sensitivity().is_nan(),
            "median with half NaN blocks should be NaN, got {}",
            result.median_sensitivity()
        );
    }

    /// Regression test for #2601: worst_k_blocks places NaN-sensitivity blocks
    /// last so they never displace finite-valued blocks from the ranking.
    #[ntest::timeout(5000)]
    #[test]
    fn test_worst_k_blocks_nan_last_2601() {
        // Blocks: [NaN, 50.0, 10.0, NaN, 30.0]
        let result = make_block_wise_result(&[f32::NAN, 50.0, 10.0, f32::NAN, 30.0]);
        let worst = result.worst_k_blocks(5);
        assert_eq!(worst.len(), 5);
        // Finite blocks descending first, then NaN
        assert_eq!(worst[0].0, 1); // block1, sensitivity 50.0
        assert_eq!(worst[1].0, 4); // block4, sensitivity 30.0
        assert_eq!(worst[2].0, 2); // block2, sensitivity 10.0
        assert!(worst[3].2.is_nan(), "4th should be NaN");
        assert!(worst[4].2.is_nan(), "5th should be NaN");
    }

    /// Regression test for #2601: worst_k_blocks(k) with k < total_blocks
    /// picks the top-k finite blocks, not NaN-contaminated ones.
    #[ntest::timeout(5000)]
    #[test]
    fn test_worst_k_blocks_nan_excluded_from_top_k_2601() {
        // Blocks: [NaN, 50.0, 10.0, NaN, 30.0], request top 3
        let result = make_block_wise_result(&[f32::NAN, 50.0, 10.0, f32::NAN, 30.0]);
        let worst = result.worst_k_blocks(3);
        assert_eq!(worst.len(), 3);
        // All 3 should be the finite blocks (NaN pushed past truncation)
        assert_eq!(worst[0].0, 1); // block1, sensitivity 50.0
        assert_eq!(worst[1].0, 4); // block4, sensitivity 30.0
        assert_eq!(worst[2].0, 2); // block2, sensitivity 10.0
        assert!(
            worst.iter().all(|w| !w.2.is_nan()),
            "top-k should contain only finite blocks"
        );
    }
}
