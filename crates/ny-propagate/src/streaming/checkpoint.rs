// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::BoundPropagation;
use crate::network::Network;
use ny_core::{NyError, Result};
use ny_tensor::{BoundedTensor, CompressedBounds};

/// Storage mode for checkpointed bounds.
#[derive(Debug, Clone)]
enum CheckpointStorage {
    /// Standard f32 storage.
    F32(Vec<(usize, BoundedTensor)>),
    /// Compressed f16 storage for reduced memory.
    F16(Vec<(usize, CompressedBounds)>),
}

/// Checkpointed bounds storage for gradient checkpointing.
///
/// Stores bounds only at checkpoint layers, enabling memory-efficient
/// CROWN propagation through recomputation. Optionally uses f16 compression
/// for additional 50% memory savings.
#[derive(Debug, Clone)]
pub struct CheckpointedBounds {
    /// Checkpoint layer indices and their bounds.
    /// Sorted by layer index for efficient lookup.
    checkpoints: CheckpointStorage,

    /// Always store input bounds (needed for recomputation from start).
    input: BoundedTensor,

    /// Total number of layers (for reference).
    num_layers: usize,

    /// Widening epsilon for f16 compression (for soundness).
    f16_widening_epsilon: f32,
}

impl CheckpointedBounds {
    /// Create new checkpointed bounds storage with f32.
    pub fn new(input: BoundedTensor, num_layers: usize) -> Self {
        Self {
            checkpoints: CheckpointStorage::F32(Vec::new()),
            input,
            num_layers,
            f16_widening_epsilon: 0.0,
        }
    }

    /// Create new checkpointed bounds storage with f16 compression.
    pub fn new_compressed(input: BoundedTensor, num_layers: usize, widening_epsilon: f32) -> Self {
        Self {
            checkpoints: CheckpointStorage::F16(Vec::new()),
            input,
            num_layers,
            f16_widening_epsilon: widening_epsilon,
        }
    }

    /// Add a checkpoint at the given layer.
    pub fn add_checkpoint(&mut self, layer_idx: usize, bounds: BoundedTensor) {
        match &mut self.checkpoints {
            CheckpointStorage::F32(checkpoints) => {
                // Keep sorted by layer index
                let pos = checkpoints
                    .binary_search_by_key(&layer_idx, |(idx, _)| *idx)
                    .unwrap_or_else(|p| p);
                checkpoints.insert(pos, (layer_idx, bounds));
            }
            CheckpointStorage::F16(checkpoints) => {
                // Compress to f16 before storing
                let mut compressed = CompressedBounds::from_bounded_tensor(&bounds);
                if self.f16_widening_epsilon > 0.0 {
                    compressed.widen_for_soundness(self.f16_widening_epsilon);
                }
                let pos = checkpoints
                    .binary_search_by_key(&layer_idx, |(idx, _)| *idx)
                    .unwrap_or_else(|p| p);
                checkpoints.insert(pos, (layer_idx, compressed));
            }
        }
    }

    /// Bounds at layer_idx by finding nearest checkpoint and recomputing if needed.
    /// Returns None if layer_idx is invalid.
    pub fn bounds_at(&self, layer_idx: usize, network: &Network) -> Result<BoundedTensor> {
        if self.num_layers == 0 || layer_idx >= self.num_layers {
            return Err(NyError::InvalidSpec(format!(
                "Layer index {} out of range (num_layers={})",
                layer_idx, self.num_layers,
            )));
        }

        // Check if we have an exact checkpoint
        match &self.checkpoints {
            CheckpointStorage::F32(checkpoints) => {
                if let Ok(pos) = checkpoints.binary_search_by_key(&layer_idx, |(idx, _)| *idx) {
                    return Ok(checkpoints[pos].1.clone());
                }
            }
            CheckpointStorage::F16(checkpoints) => {
                if let Ok(pos) = checkpoints.binary_search_by_key(&layer_idx, |(idx, _)| *idx) {
                    return checkpoints[pos].1.to_bounded_tensor();
                }
            }
        }

        // Find nearest checkpoint before this layer
        let (start_idx, start_bounds) = self.find_nearest_checkpoint_before(layer_idx)?;

        // Recompute forward from checkpoint to target layer
        self.recompute_forward(network, start_idx, layer_idx, start_bounds)
    }

    /// Find the nearest checkpoint at or before the given layer.
    /// Returns (checkpoint_layer, checkpoint_bounds).
    /// If no checkpoint exists before `layer_idx`, returns (-1, input_bounds).
    ///
    /// Uses binary search (O(log K)) since checkpoints are stored sorted by layer index.
    pub(crate) fn find_nearest_checkpoint_before(
        &self,
        layer_idx: usize,
    ) -> Result<(i64, BoundedTensor)> {
        match &self.checkpoints {
            CheckpointStorage::F32(checkpoints) => {
                // partition_point: count of elements where idx <= layer_idx
                let pos = checkpoints.partition_point(|(idx, _)| *idx <= layer_idx);
                if pos == 0 {
                    Ok((-1, self.input.clone()))
                } else {
                    let (idx, bounds) = &checkpoints[pos - 1];
                    Ok((*idx as i64, bounds.clone()))
                }
            }
            CheckpointStorage::F16(checkpoints) => {
                let pos = checkpoints.partition_point(|(idx, _)| *idx <= layer_idx);
                if pos == 0 {
                    Ok((-1, self.input.clone()))
                } else {
                    let (idx, compressed) = &checkpoints[pos - 1];
                    let bounds = compressed.to_bounded_tensor().map_err(|e| {
                        NyError::InvalidSpec(format!(
                            "Checkpoint decompression failed at layer {}: {}",
                            idx, e
                        ))
                    })?;
                    Ok((*idx as i64, bounds))
                }
            }
        }
    }

    /// Recompute forward from start_idx to target_idx.
    /// start_idx = -1 means start from input.
    fn recompute_forward(
        &self,
        network: &Network,
        start_idx: i64,
        target_idx: usize,
        start_bounds: BoundedTensor,
    ) -> Result<BoundedTensor> {
        let mut current = start_bounds;

        // start_idx is the output of layer start_idx (or input if -1)
        // We need to propagate through layers (start_idx+1) to target_idx
        if start_idx < -1 {
            return Err(NyError::InvalidSpec(format!(
                "start_idx must be >= -1, got {start_idx}"
            )));
        }
        // SAFETY: start_idx >= -1, so start_idx + 1 >= 0, fits in usize.
        let first_layer = (start_idx + 1) as usize;

        for i in first_layer..=target_idx {
            let layer = network
                .layers
                .get(i)
                .ok_or_else(|| NyError::InvalidSpec(format!("Layer {} not found", i)))?;

            current = layer
                .propagate_ibp(&current)
                .map_err(|e| NyError::LayerError {
                    layer_index: i,
                    layer_type: layer.layer_type().to_string(),
                    source: Box::new(e),
                })?;
        }

        Ok(current)
    }

    /// Number of checkpoints stored.
    pub fn num_checkpoints(&self) -> usize {
        match &self.checkpoints {
            CheckpointStorage::F32(checkpoints) => checkpoints.len(),
            CheckpointStorage::F16(checkpoints) => checkpoints.len(),
        }
    }

    /// Estimated memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let input_size = self.input.lower().len() * 4 * 2; // f32 lower + upper
        let checkpoint_size: usize = match &self.checkpoints {
            CheckpointStorage::F32(checkpoints) => checkpoints
                .iter()
                .map(|(_, b)| b.lower().len() * 4 * 2) // f32: 4 bytes per element
                .sum(),
            CheckpointStorage::F16(checkpoints) => checkpoints
                .iter()
                .map(|(_, b)| b.len() * 2 * 2) // f16: 2 bytes per element
                .sum(),
        };
        input_size + checkpoint_size
    }

    /// Check if using f16 compression.
    pub fn is_compressed(&self) -> bool {
        matches!(self.checkpoints, CheckpointStorage::F16(_))
    }

    /// Get compression statistics if using f16.
    /// Returns (total_f16_bytes, equivalent_f32_bytes, ratio) or None if not compressed.
    pub fn compression_stats(&self) -> Option<(usize, usize, f32)> {
        if let CheckpointStorage::F16(checkpoints) = &self.checkpoints {
            let f16_bytes: usize = checkpoints.iter().map(|(_, b)| b.memory_bytes()).sum();
            let f32_bytes: usize = checkpoints.iter().map(|(_, b)| b.len() * 4 * 2).sum();
            let ratio = if f32_bytes > 0 {
                f16_bytes as f32 / f32_bytes as f32
            } else {
                1.0
            };
            Some((f16_bytes, f32_bytes, ratio))
        } else {
            None
        }
    }

    /// Get the last checkpoint's bounds (output bounds).
    /// Returns `Ok(None)` if no checkpoints exist, `Ok(Some(bounds))` if found,
    /// or `Err(...)` if f16 decompression fails.
    pub fn last_checkpoint(&self) -> Result<Option<BoundedTensor>> {
        match &self.checkpoints {
            CheckpointStorage::F32(checkpoints) => Ok(checkpoints.last().map(|(_, b)| b.clone())),
            CheckpointStorage::F16(checkpoints) => match checkpoints.last() {
                None => Ok(None),
                Some((idx, compressed)) => {
                    let bounds = compressed.to_bounded_tensor().map_err(|e| {
                        NyError::InvalidSpec(format!(
                            "Last checkpoint decompression failed at layer {}: {}",
                            idx, e
                        ))
                    })?;
                    Ok(Some(bounds))
                }
            },
        }
    }
}
