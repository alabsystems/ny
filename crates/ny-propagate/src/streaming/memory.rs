// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Calculate estimated memory savings from streaming.
///
/// Returns (original_memory_bytes, streaming_memory_bytes, savings_percent).
pub fn estimate_memory_savings(
    num_layers: usize,
    tensor_elements: usize,
    checkpoint_interval: usize,
) -> (usize, usize, f32) {
    let interval = checkpoint_interval.max(1);
    let bytes_per_tensor = tensor_elements * 4 * 2; // f32 lower + upper

    // Original: store all layers
    let original = num_layers * bytes_per_tensor;

    // Streaming: store checkpoints + input
    let num_checkpoints = num_layers.div_ceil(interval);
    let streaming = (num_checkpoints + 1) * bytes_per_tensor;

    let savings = if original > 0 {
        100.0 * (1.0 - streaming as f32 / original as f32)
    } else {
        0.0
    };

    (original, streaming, savings)
}
