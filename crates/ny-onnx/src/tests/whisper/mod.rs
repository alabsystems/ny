// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Whisper test coverage.
//!
//! Tests backed by the generated `whisper_tiny_encoder.onnx` artifact belong
//! to the explicit `external-whisper` conformance lane. Generate it with
//! `scripts/export_whisper_encoder.py`, then run
//! `cargo run -p ny-onnx --bin ny_onnx_conformance -- whisper`. Missing assets
//! fail that lane; fixture-free synthetic tests remain in the default suite.

mod helpers;

mod attention;
mod attention_context_fallback;
mod attention_context_matrix;
mod attention_stage_localization;
mod basic;
#[cfg(feature = "benchmarks")]
mod block_ibp_scaling;
mod compositional;
#[cfg(feature = "benchmarks")]
mod compositional_fixture_3450;
mod graph;
mod structure;
mod zonotope;
