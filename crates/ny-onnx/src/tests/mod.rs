// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#[cfg(any(feature = "benchmarks", feature = "external-whisper"))]
use crate::MultiBlockConfig;
use crate::{
    load_decoder, load_onnx, load_whisper, onnx_proto, AttributeValue, DataType,
    GraphNetworkOptions, LayerSpec, Network, OnnxModel, PropagateNetworkOptions, TensorSpec,
    WeightStore, WhisperEncoderStructure, WhisperModel,
};

mod core;
mod fixtures;
#[cfg_attr(not(feature = "external-whisper"), allow(dead_code, unused_imports))]
#[path = "whisper/mod.rs"]
mod whisper;
