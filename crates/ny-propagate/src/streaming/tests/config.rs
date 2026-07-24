// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::streaming::*;

#[ntest::timeout(5000)]
#[test]
fn test_streaming_config_defaults() {
    let config = StreamingConfig::default();
    assert_eq!(config.checkpoint_interval, 10);
    assert!(!config.use_f16_checkpoints);
    assert!((config.f16_widening_epsilon - 0.001).abs() < f32::EPSILON);
}

#[ntest::timeout(5000)]
#[test]
fn test_streaming_config_min_memory() {
    let config = StreamingConfig::min_memory();
    assert_eq!(config.checkpoint_interval, 50);
    assert!(config.use_f16_checkpoints);
}

#[ntest::timeout(5000)]
#[test]
fn test_f16_compression_config() {
    let config = StreamingConfig::with_f16_compression();
    assert!(config.use_f16_checkpoints);
    assert_eq!(config.checkpoint_interval, 10);

    let config = StreamingConfig::min_memory();
    assert!(config.use_f16_checkpoints);
    assert_eq!(config.checkpoint_interval, 50);
}
