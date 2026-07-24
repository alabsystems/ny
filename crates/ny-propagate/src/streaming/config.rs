// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Configuration for streaming/checkpointed computation.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Number of layers between checkpoints.
    /// Smaller = more memory, less recomputation.
    /// Larger = less memory, more recomputation.
    /// Default: 10 (stores every 10th layer's bounds).
    pub checkpoint_interval: usize,

    /// Use f16 (half precision) for checkpoint storage.
    /// Provides additional 50% memory reduction at cost of precision.
    /// Default: false. Use for very large models or memory-constrained environments.
    pub use_f16_checkpoints: bool,

    /// Relative epsilon for sound bound widening when using f16.
    /// After decompression, bounds are widened by this factor to ensure soundness.
    /// Default: 0.001 (0.1%). Set to 0.0 to disable widening.
    pub f16_widening_epsilon: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: 10,
            use_f16_checkpoints: false,
            f16_widening_epsilon: 0.001,
        }
    }
}

impl StreamingConfig {
    /// Create config optimized for minimum memory usage.
    /// Uses large checkpoint interval + f16 storage for maximum memory savings.
    pub fn min_memory() -> Self {
        Self {
            checkpoint_interval: 50,
            use_f16_checkpoints: true,
            f16_widening_epsilon: 0.001,
        }
    }

    /// Create config with f16 compression enabled.
    /// Provides ~50% additional memory reduction vs f32 checkpoints.
    pub fn with_f16_compression() -> Self {
        Self {
            checkpoint_interval: 10,
            use_f16_checkpoints: true,
            f16_widening_epsilon: 0.001,
        }
    }
}
