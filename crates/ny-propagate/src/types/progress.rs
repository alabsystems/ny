// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Progress tracking types for verification.

use std::time::Duration;

/// Progress information reported during block-wise verification.
#[derive(Debug, Clone)]
pub struct BlockProgress {
    /// Current block index being processed (0-based).
    pub block_index: usize,
    /// Total number of blocks to process.
    pub total_blocks: usize,
    /// Name of the current block.
    pub block_name: String,
    /// Elapsed time since verification started.
    pub elapsed: Duration,
    /// Current max sensitivity seen so far.
    pub current_max_sensitivity: f32,
    /// Number of degraded blocks so far.
    pub degraded_so_far: usize,
}

impl BlockProgress {
    /// Progress as a fraction (0.0 to 1.0).
    pub fn fraction(&self) -> f32 {
        if self.total_blocks == 0 {
            1.0
        } else {
            // The callback is invoked after the block completes.
            (self.block_index + 1) as f32 / self.total_blocks as f32
        }
    }

    /// Estimated time remaining based on current progress.
    pub fn estimated_remaining(&self) -> Duration {
        let fraction = self.fraction();
        if fraction > 0.0 && fraction < 1.0 {
            let elapsed_secs = self.elapsed.as_secs_f64();
            let total_estimated = elapsed_secs / fraction as f64;
            let remaining = total_estimated - elapsed_secs;
            Duration::from_secs_f64(remaining.max(0.0))
        } else {
            Duration::ZERO
        }
    }

    /// Estimated time remaining based on average block time so far.
    pub fn eta(&self) -> Duration {
        let completed = self.block_index + 1;
        if completed == 0 || completed >= self.total_blocks {
            return Duration::ZERO;
        }
        let avg_per_block = self.elapsed.as_secs_f64() / completed as f64;
        let remaining = self.total_blocks - completed;
        Duration::from_secs_f64(avg_per_block * remaining as f64)
    }
}

/// Progress information reported during layer-by-layer verification within a block.
#[derive(Debug, Clone)]
pub struct LayerProgress {
    /// Current node index being processed (0-based).
    pub node_index: usize,
    /// Total number of nodes to process.
    pub total_nodes: usize,
    /// Name of the current node.
    pub node_name: String,
    /// Layer type of the current node.
    pub layer_type: String,
    /// Elapsed time since verification started.
    pub elapsed: Duration,
    /// Current max sensitivity seen so far.
    pub current_max_sensitivity: f32,
    /// Number of degraded nodes so far.
    pub degraded_so_far: usize,
}

impl LayerProgress {
    /// Progress as a fraction complete (0.0 to 1.0).
    pub fn fraction(&self) -> f32 {
        if self.total_nodes == 0 {
            1.0
        } else {
            (self.node_index + 1) as f32 / self.total_nodes as f32
        }
    }

    /// Estimated time remaining based on current progress.
    pub fn estimated_remaining(&self) -> Duration {
        let fraction = self.fraction();
        if fraction > 0.0 && fraction < 1.0 {
            let elapsed_secs = self.elapsed.as_secs_f64();
            let total_estimated = elapsed_secs / fraction as f64;
            let remaining = total_estimated - elapsed_secs;
            Duration::from_secs_f64(remaining.max(0.0))
        } else {
            Duration::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_progress_fraction_normal() {
        let progress = BlockProgress {
            block_index: 2, // Completed blocks 0, 1, 2 (3 total)
            total_blocks: 10,
            block_name: "block2".to_string(),
            elapsed: Duration::from_secs(30),
            current_max_sensitivity: 5.0,
            degraded_so_far: 0,
        };
        // After completing block 2, we have 3/10 complete
        assert!((progress.fraction() - 0.3).abs() < 1e-6);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_progress_fraction_zero_total() {
        let progress = BlockProgress {
            block_index: 0,
            total_blocks: 0,
            block_name: String::new(),
            elapsed: Duration::ZERO,
            current_max_sensitivity: 0.0,
            degraded_so_far: 0,
        };
        assert!((progress.fraction() - 1.0).abs() < 1e-6);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_progress_eta() {
        let progress = BlockProgress {
            block_index: 4, // 5 blocks done
            total_blocks: 10,
            block_name: "block4".to_string(),
            elapsed: Duration::from_secs(50), // 10 sec per block
            current_max_sensitivity: 5.0,
            degraded_so_far: 0,
        };
        // 5 remaining blocks * 10 sec = 50 sec
        let eta = progress.eta();
        assert!((eta.as_secs_f64() - 50.0).abs() < 0.1);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_progress_eta_complete() {
        let progress = BlockProgress {
            block_index: 9,
            total_blocks: 10,
            block_name: "block9".to_string(),
            elapsed: Duration::from_secs(100),
            current_max_sensitivity: 5.0,
            degraded_so_far: 0,
        };
        assert_eq!(progress.eta(), Duration::ZERO);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_block_progress_estimated_remaining() {
        let progress = BlockProgress {
            block_index: 4, // 5 blocks done = 50%
            total_blocks: 10,
            block_name: "block4".to_string(),
            elapsed: Duration::from_secs(50),
            current_max_sensitivity: 5.0,
            degraded_so_far: 0,
        };
        let remaining = progress.estimated_remaining();
        // 50% done in 50 sec, so 50 sec remaining
        assert!((remaining.as_secs_f64() - 50.0).abs() < 0.1);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_layer_progress_fraction() {
        let progress = LayerProgress {
            node_index: 24, // 25 nodes done
            total_nodes: 100,
            node_name: "layer24".to_string(),
            layer_type: "Linear".to_string(),
            elapsed: Duration::from_secs(25),
            current_max_sensitivity: 3.0,
            degraded_so_far: 0,
        };
        assert!((progress.fraction() - 0.25).abs() < 1e-6);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_layer_progress_fraction_zero_total() {
        let progress = LayerProgress {
            node_index: 0,
            total_nodes: 0,
            node_name: String::new(),
            layer_type: String::new(),
            elapsed: Duration::ZERO,
            current_max_sensitivity: 0.0,
            degraded_so_far: 0,
        };
        assert!((progress.fraction() - 1.0).abs() < 1e-6);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_layer_progress_estimated_remaining() {
        let progress = LayerProgress {
            node_index: 49, // 50 nodes done = 50%
            total_nodes: 100,
            node_name: "layer49".to_string(),
            layer_type: "ReLU".to_string(),
            elapsed: Duration::from_secs(50),
            current_max_sensitivity: 2.0,
            degraded_so_far: 0,
        };
        let remaining = progress.estimated_remaining();
        assert!((remaining.as_secs_f64() - 50.0).abs() < 0.1);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_layer_progress_estimated_remaining_complete() {
        let progress = LayerProgress {
            node_index: 99,
            total_nodes: 100,
            node_name: "layer99".to_string(),
            layer_type: "Output".to_string(),
            elapsed: Duration::from_secs(100),
            current_max_sensitivity: 2.0,
            degraded_so_far: 0,
        };
        assert_eq!(progress.estimated_remaining(), Duration::ZERO);
    }
}
