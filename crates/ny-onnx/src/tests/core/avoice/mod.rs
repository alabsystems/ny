// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Avoice model-family test suite.
//!
//! Each model family (duration predictor, kokoro vocoder, speaker encoder,
//! talker attention) lives in its own directory-backed sub-module. Shared
//! helpers are consolidated in `common.rs`.
//!
//! # Wall-clock budget policy
//!
//! Every `ntest::timeout` in this tree is a wall-clock watchdog sized from
//! *release-mode* measurements on the original development hardware. Debug
//! builds run this workload 2-30x slower, so asserting the same wall-clock
//! numbers in debug produces guaranteed reds that say nothing about
//! correctness (measured 2026-07-19 on an M5 Max: the crossfade prefix-IBP
//! fixture alone blows its 120s budget solo in debug but builds in well under
//! that in release). The budgets are therefore asserted only under
//! `--release` via `#[cfg_attr(not(debug_assertions), ntest::timeout(..))]`;
//! debug runs execute every correctness assertion unconditionally and are
//! simply allowed to take the time they take.

use super::fixtures::{load_avoice_contract, require_test_model_with_hint, AVOICE_TEST_MODEL_HINT};
use super::{load_onnx, DataType, OnnxModel, TensorSpec};
use ny_core::LayerType;
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::sync::OnceLock;

mod common;
mod duration_predictor;
mod kokoro_vocoder;
mod speaker_encoder;
mod talker_attention;
mod training_signal_support;
mod verifier_smoke;
#[cfg(feature = "ort")]
mod vocoder_speaker_bridge;
