// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint types for resumable verification.

use super::block_bounds::{BlockBoundsInfo, BlockWiseResult};
use super::helpers::chrono_lite_now;
use ny_core::{nan_propagating_max, NyError, Result};
use serde::{Deserialize, Serialize};

/// Checkpoint for resumable block-wise verification.
///
/// Allows long-running verification to be interrupted and resumed without
/// losing progress. Serialized to JSON for human readability and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCheckpoint {
    /// Checkpoint format version for compatibility.
    pub version: u32,

    /// Model file path (for validation).
    pub model_path: std::path::PathBuf,

    /// SHA256 hash of the first 64KB of the model file (lowercase hex, 64 chars).
    pub model_hash: String,

    /// Input epsilon used for verification.
    pub epsilon: f32,

    /// Verification method (ibp, crown, alpha, beta).
    pub method: String,

    /// Compute backend (cpu, wgpu).
    pub backend: String,

    /// Timestamp when verification started (ISO 8601).
    pub start_time: String,

    /// Timestamp of this checkpoint (ISO 8601).
    pub checkpoint_time: String,

    /// Total elapsed time in milliseconds (excluding pauses).
    pub elapsed_ms: u64,

    /// Completed blocks with full results.
    pub completed_blocks: Vec<BlockBoundsInfo>,

    /// Maximum sensitivity across completed blocks.
    pub max_sensitivity: f32,

    /// Number of degraded blocks so far.
    pub degraded_blocks: usize,

    /// Total number of blocks in the model.
    pub total_blocks: usize,

    /// Next block index to process (resume point).
    pub next_block_index: usize,
}

impl VerificationCheckpoint {
    /// Current checkpoint format version.
    pub const VERSION: u32 = 1;

    /// Create a new checkpoint at the start of verification.
    pub fn new(
        model_path: std::path::PathBuf,
        model_hash: String,
        epsilon: f32,
        method: &str,
        backend: &str,
        total_blocks: usize,
    ) -> Self {
        let now = chrono_lite_now();
        Self {
            version: Self::VERSION,
            model_path,
            model_hash,
            epsilon,
            method: method.to_string(),
            backend: backend.to_string(),
            start_time: now.clone(),
            checkpoint_time: now,
            elapsed_ms: 0,
            completed_blocks: Vec::new(),
            max_sensitivity: 0.0,
            degraded_blocks: 0,
            total_blocks,
            next_block_index: 0,
        }
    }

    /// Update checkpoint after completing a block.
    pub fn update(&mut self, block: BlockBoundsInfo, elapsed_ms: u64) {
        self.max_sensitivity = nan_propagating_max(self.max_sensitivity, block.sensitivity);
        if block.degraded {
            self.degraded_blocks += 1;
        }
        self.next_block_index = block.block_index + 1;
        self.completed_blocks.push(block);
        self.elapsed_ms = elapsed_ms;
        self.checkpoint_time = chrono_lite_now();
    }

    /// Save checkpoint to file atomically.
    ///
    /// Uses write-to-temp-then-rename pattern to ensure the checkpoint file
    /// is never corrupted, even if a crash occurs during write. This is critical
    /// for multi-hour verification runs.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| NyError::InvalidSpec(format!("Failed to serialize checkpoint: {}", e)))?;

        // Create temp file in same directory (ensures same filesystem for atomic rename)
        let temp_path = path.with_extension("json.tmp");

        // Write to temp file
        let mut file = std::fs::File::create(&temp_path).map_err(|e| {
            NyError::InvalidSpec(format!("Failed to create temp checkpoint file: {}", e))
        })?;

        file.write_all(json.as_bytes())
            .map_err(|e| NyError::InvalidSpec(format!("Failed to write temp checkpoint: {}", e)))?;

        // Sync to disk before rename (ensures data is durable)
        file.sync_all().map_err(|e| {
            NyError::InvalidSpec(format!("Failed to sync checkpoint to disk: {}", e))
        })?;

        // Drop file handle before rename
        drop(file);

        // Atomic rename (POSIX guarantees atomicity on same filesystem)
        std::fs::rename(&temp_path, path)
            .map_err(|e| NyError::InvalidSpec(format!("Failed to rename checkpoint: {}", e)))?;

        Ok(())
    }

    /// Load checkpoint from file.
    ///
    /// Also cleans up any stale temp files from interrupted saves.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        // Clean up stale temp file if it exists (from interrupted save)
        let temp_path = path.with_extension("json.tmp");
        if temp_path.exists() {
            // Best effort cleanup - ignore errors
            let _ = std::fs::remove_file(&temp_path);
        }

        let json = std::fs::read_to_string(path)
            .map_err(|e| NyError::InvalidSpec(format!("Failed to read checkpoint: {}", e)))?;
        let checkpoint: Self = serde_json::from_str(&json)
            .map_err(|e| NyError::InvalidSpec(format!("Failed to parse checkpoint: {}", e)))?;

        if checkpoint.version != Self::VERSION {
            return Err(NyError::InvalidSpec(format!(
                "Checkpoint version mismatch: expected {}, found {}",
                Self::VERSION,
                checkpoint.version
            )));
        }

        Ok(checkpoint)
    }

    /// Validate checkpoint matches current verification config.
    pub fn validate(
        &self,
        model_path: &std::path::Path,
        model_hash: &str,
        epsilon: f32,
        method: &str,
        backend: &str,
    ) -> Result<()> {
        if self.model_path != model_path {
            return Err(NyError::InvalidSpec(format!(
                "Checkpoint model path mismatch: checkpoint has {}, current is {}",
                self.model_path.display(),
                model_path.display()
            )));
        }
        if self.model_hash != model_hash {
            return Err(NyError::InvalidSpec(
                "Checkpoint model hash mismatch: model file has changed since checkpoint"
                    .to_string(),
            ));
        }
        if (self.epsilon - epsilon).abs() > 1e-9 {
            return Err(NyError::InvalidSpec(format!(
                "Checkpoint epsilon mismatch: checkpoint has {}, current is {}",
                self.epsilon, epsilon
            )));
        }
        if self.method != method {
            return Err(NyError::InvalidSpec(format!(
                "Checkpoint method mismatch: checkpoint has {}, current is {}",
                self.method, method
            )));
        }
        if self.backend != backend {
            return Err(NyError::InvalidSpec(format!(
                "Checkpoint backend mismatch: checkpoint has {}, current is {}",
                self.backend, backend
            )));
        }
        Ok(())
    }

    /// Check if verification is complete.
    pub fn is_complete(&self) -> bool {
        self.next_block_index >= self.total_blocks
    }

    /// Build final result from completed checkpoint.
    ///
    /// Returns `Err` if the checkpoint is not complete. Checks both
    /// `next_block_index` (resume pointer) and `completed_blocks.len()` (actual
    /// count) to guard against out-of-order `update()` calls that could advance
    /// `next_block_index` past `total_blocks` without processing all
    /// intermediate blocks. (#2808)
    pub fn into_result(self) -> Result<BlockWiseResult> {
        if !self.is_complete() {
            return Err(NyError::InvalidSpec(format!(
                "into_result() called on incomplete checkpoint: \
                 next_block_index={}, total_blocks={}",
                self.next_block_index, self.total_blocks,
            )));
        }
        if self.completed_blocks.len() != self.total_blocks {
            return Err(NyError::InvalidSpec(format!(
                "into_result() block count mismatch: \
                 completed_blocks.len()={} != total_blocks={} \
                 (possible out-of-order update)",
                self.completed_blocks.len(),
                self.total_blocks,
            )));
        }
        Ok(BlockWiseResult {
            blocks: self.completed_blocks,
            block_epsilon: self.epsilon,
            total_blocks: self.total_blocks,
            max_sensitivity: self.max_sensitivity,
            degraded_blocks: self.degraded_blocks,
        })
    }
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
