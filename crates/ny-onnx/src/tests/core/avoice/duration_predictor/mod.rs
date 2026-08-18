// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kokoro duration predictor verification tests for #3497.
//!
//! The Kokoro duration predictor outputs `duration_logits [B, T, 50]` where
//! the 50 channels are independent Bernoulli bins. Expected duration is
//! `sigmoid(logits).sum(-1)`, NOT a categorical softmax expectation. The
//! production avoice path then divides by `speed` and clamps the continuous
//! duration counts into `[1, 50]`.
//!
//! Three test categories:
//! 1. **Synthetic proof head tests** — verify both the raw sigmoid+sum
//!    expected-duration formula and the production `duration_to_counts`
//!    speed/clamp semantics on hand-crafted logits, no ONNX model needed.
//! 2. **Feed-forward surrogate pipeline** — load `kokoro_duration_predictor_surrogate.onnx`,
//!    convert to graph, run IBP, apply the proof head, verify positive
//!    expected durations, then check the production count surface.
//! 3. **BiLSTM sequence-output admission** — preserve refusal for the real
//!    Kokoro-style full-sequence path until lowering implements ONNX's exact
//!    four-dimensional bidirectional Y layout. Final-state Y_h/Y_c lowering is
//!    exercised separately by the loader tests.
//!
//! Sources:
//! - designs/2026-03-11-avoice-phase1-onnx-execution.md (section 4)
//! - `./avoice/scripts/export_kokoro_onnx.py` (output shape)
//! - `./avoice/crates/avoice-tts/src/kokoro/model_ops.rs`

use super::*;
use crate::tests::fixtures::require_test_model;
use ndarray::{ArrayD, IxDyn};
use ny_propagate::layers::{ReduceSumLayer, SigmoidLayer};
use ny_propagate::GraphNetwork;

mod bilstm;
mod proof_head;
mod real_export;
mod surrogate;
pub(crate) mod verifier_smoke;
