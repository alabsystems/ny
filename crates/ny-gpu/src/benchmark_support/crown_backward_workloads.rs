// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared representative workload shapes for the #3397 GPU CROWN backward
//! benchmark harness, measurement example, and timing tests.

/// Deterministic LCG random f32 for reproducible benchmark/test data.
/// Uses the same algorithm across all GPU CROWN benchmark consumers.
pub fn bench_rng_f32(seed: &mut u64, scale: f32) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * scale
}

pub type ConvBenchSpec = (
    usize,
    usize,
    usize,
    (usize, usize),
    (usize, usize),
    (usize, usize),
);

pub const METAROOM_CASE_NAME: &str = "metaroom_6cnn_ry_like";
pub const METAROOM_INPUT_SHAPE: [usize; 3] = [3, 32, 56];
pub const METAROOM_HIDDEN_DIM: usize = 256;
pub const METAROOM_OUTPUT_DIM: usize = 20;
pub const METAROOM_CONV_SPECS: [ConvBenchSpec; 4] = [
    (32, 3, 3, (1, 1), (1, 1), (32, 56)),
    (32, 32, 3, (1, 1), (1, 1), (32, 56)),
    (64, 32, 3, (2, 2), (1, 1), (32, 56)),
    (64, 64, 3, (1, 1), (1, 1), (16, 28)),
];

pub const SOUNDNESSBENCH_CASE_NAME: &str = "soundnessbench_exact_like";
pub const SOUNDNESSBENCH_INPUT_DIM: usize = 128;
pub const SOUNDNESSBENCH_RESHAPE_SHAPE: [usize; 3] = [3, 64, 64];
pub const SOUNDNESSBENCH_OUTPUT_DIM: usize = 384;
pub const SOUNDNESSBENCH_CONV_SPECS: [ConvBenchSpec; 6] = [
    (24, 3, 1, (1, 1), (0, 0), (64, 64)),
    (24, 24, 3, (1, 1), (1, 1), (64, 64)),
    (24, 24, 1, (2, 2), (0, 0), (64, 64)),
    (24, 24, 1, (2, 2), (0, 0), (32, 32)),
    (24, 24, 1, (2, 2), (0, 0), (16, 16)),
    (24, 24, 1, (2, 2), (0, 0), (8, 8)),
];

pub const fn shape_product3(shape: [usize; 3]) -> usize {
    shape[0] * shape[1] * shape[2]
}

pub fn conv_output_dim(spec: ConvBenchSpec) -> usize {
    let (out_channels, _, kernel, (stride_h, stride_w), (pad_h, pad_w), (input_h, input_w)) = spec;
    let out_h = (input_h + 2 * pad_h - kernel) / stride_h + 1;
    let out_w = (input_w + 2 * pad_w - kernel) / stride_w + 1;
    out_channels * out_h * out_w
}
