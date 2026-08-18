// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Command handlers for the ny CLI.
//!
//! The handlers are grouped by responsibility rather than by clap definitions:
//! - `analysis`: diff, quantization, profiling, and related analysis commands
//! - `inspect` / `inspect_model`: model inspection, comparison, and format introspection
//! - `verify`: the main incomplete-verification entry points plus helper modules
//! - `beta_crown`: complete verification, branching, and MIP/SMT-backed helpers
//! - `bench`, `bench_acasxu`, `bench_vnncomp`: benchmarking and competition probes
//! - `weights` and `whisper`: specialized model-analysis and Whisper/TTS workflows
//! - `backend`: backend selection helpers shared across command handlers
//! - `json_error`: schema-stable CLI error payloads
//! - `gt`: geometric ground-truth utilities (`ny gt eval`/`ny gt verify` over
//!   `.gt.json` sidecars and ground-truth VNN-LIB dual-network properties)
//! - `vnncomp`: native VNN-COMP `run_instance.sh` flow (preset auto-load, timeout
//!   tiering, β-CROWN invocation, verdict translation, RESULTS_FILE writing)
//! - `vnncomp_benchmarks`: VNN-COMP benchmark workflow command routing
//! - `vnncomp_sweep`: `ny benchmarks run` — sweep a corpus to an official-format results.csv
//! - `vnncomp_score`: compare result banks and model separate 2025/2026 track scores
//! - `vnncomp_reseed`: fail-closed path-rename metamorphic checks
//! - `vnncomp_submit`: VNN-COMP harness validation and submission packaging

// Command-handler modules.
pub(crate) mod analysis;
pub(crate) mod backend;
pub(crate) mod bench;
pub(crate) mod bench_acasxu;
pub(crate) mod bench_vnncomp;
pub(crate) mod beta_crown;
pub(crate) mod cgan_status;
pub(crate) mod coupled_delta;
pub(crate) mod coverage;
// Qualification milestone: compiled and tested under `mip`, deliberately
// unreachable from commands/verdicts until the full-network/replay gate lands.
#[cfg(feature = "mip")]
#[allow(dead_code)]
pub(crate) mod cz_metaroom_unwired;
// Default-dark real-model qualification machinery. Its public probe reports
// remain verdict-neutral; the beta-crown cGAN input-leaf module is the sole
// property-bound authority layer for the private scalar leaf-row API.
#[cfg(feature = "mip")]
#[allow(dead_code)]
pub(crate) mod cz_cgan_sequential_unwired;
pub(crate) mod gt;
pub(crate) mod inspect;
pub(crate) mod inspect_model;
pub(crate) mod json_error;
pub(crate) mod lipschitz;
pub(crate) mod margin_row_bab;
pub(crate) mod relational_equiv;
pub(crate) mod terminal_peel;
#[cfg(test)]
mod tests;
pub(crate) mod tll_structure;
pub(crate) mod tutorial;
pub(crate) mod verify;
pub(crate) mod vnncomp;
pub(crate) mod vnncomp_2025_tracks;
pub(crate) mod vnncomp_2026_tracks;
pub(crate) mod vnncomp_benchmarks;
pub(crate) mod vnncomp_late_submit;
pub(crate) mod vnncomp_matrix;
pub(crate) mod vnncomp_plan;
pub(crate) mod vnncomp_reseed;
pub(crate) mod vnncomp_score;
pub(crate) mod vnncomp_submit;
pub(crate) mod vnncomp_sweep;
pub(crate) mod weights;
pub(crate) mod whisper;

pub(crate) use json_error::{find_json_cli_error, JsonCliError};

/// Shape-inference execution backend for CLI-owned model loads.
///
/// Real `ny` processes delegate ONNX Runtime shape inference to a child
/// process (`current_exe()` re-invoked with the hidden
/// [`ny_onnx::SHAPE_INFER_SUBCOMMAND`] entry served in `main.rs`): malformed
/// models can make ORT's native layer abort or fault despite Rust panic
/// recovery. In a child, that failure is just a non-zero exit status and the
/// load degrades to the sound no-inferred-shapes fallback.
///
/// Unit-test builds keep the historical in-process backend: under `cargo test`
/// `current_exe()` is the libtest harness, which cannot serve the hidden
/// subcommand (it would treat it as a test-name filter and print harness
/// chatter on stdout), so delegating there would silently drop shape inference
/// for every in-crate test. Integration tests spawn the real `ny` binary,
/// which is compiled without `cfg(test)` and exercises the subprocess backend.
pub(crate) fn cli_shape_infer_backend() -> ny_onnx::ShapeInferBackend {
    #[cfg(test)]
    {
        ny_onnx::ShapeInferBackend::InProcess
    }
    #[cfg(not(test))]
    {
        match std::env::current_exe() {
            Ok(exe) => ny_onnx::ShapeInferBackend::Subprocess { exe },
            Err(err) => {
                // No subprocess was attempted, so this is not a subprocess
                // failure: keep the historical in-process behavior.
                tracing::warn!(
                    "current_exe() unavailable ({err}); ORT shape inference stays in-process"
                );
                ny_onnx::ShapeInferBackend::InProcess
            }
        }
    }
}
