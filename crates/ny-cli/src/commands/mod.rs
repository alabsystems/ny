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
pub(crate) mod bnn_falsification_policy;
// #cctsdb-subbox: spec-aware two-sub-box disjunction for patch-attack models
// whose graph cannot be built as one unit. Reached ONLY from the vnncomp arm
// where the model already failed to load.
pub(crate) mod cctsdb_subbox;
pub(crate) mod coupled_delta;
pub(crate) mod coverage;
// #cgan-child-zono: real-model soundness pins for the shared certified
// zonotope core (the four ConvT cGANs, Z1 parity, sat leaves, refusals).
#[cfg(test)]
mod cgan_child_zono_tests;
// #cgan-child-zono: the dense core against ny-mip's exact-dyadic CZ oracle.
#[cfg(all(test, feature = "mip"))]
mod cgan_child_zono_oracle_tests;
// Qualification milestone: compiled and tested under `mip`. Historically
// deliberately unreachable from commands/verdicts; since #cz-hybrid its
// `qualification` submodule exposes one wired production entry, reached only
// through `cz_metaroom_wired` below.
#[cfg(feature = "mip")]
#[allow(dead_code)]
pub(crate) mod cz_metaroom_unwired;
// #cz-hybrid: the wired, preset-gated CZ+Box verdict lane for the metaroom
// spec_idx_119 instance allowlist (see `configs/vnncomp25/metaroom_2023.yaml`).
#[cfg(feature = "mip")]
pub(crate) mod cz_metaroom_wired;
// Default-dark real-model qualification machinery. Its public probe reports
// remain verdict-neutral. The beta-crown cGAN input-leaf module that used to
// be its property-bound authority layer was category-name-keyed and
// env-dark (`NY_CGAN_INPUT_LEAF`), and was removed by #name-blind-routing; the
// probes stay reachable from the `cz_cgan_*` examples and their own tests.
#[cfg(feature = "mip")]
#[allow(dead_code)]
pub(crate) mod cz_cgan_sequential_unwired;
/// #graph-bab-cover-ledger: the graph lane's instance door at publication.
pub(crate) mod graph_cover_publish;
pub(crate) mod gt;
pub(crate) mod head_verify;
pub(crate) mod inspect;
pub(crate) mod inspect_model;
pub(crate) mod json_error;
pub(crate) mod lane_allocation;
pub(crate) mod lane_schedule;
pub(crate) mod lipschitz;
pub(crate) mod margin_row_bab;
pub(crate) mod original_property_census;
pub(crate) mod original_source_census;
pub(crate) mod prepared_root_publication;
pub(crate) mod prepared_root_route;
pub(crate) mod prepared_root_source;
/// ENGINE_PROGRAM §1.7 B1: the scored entry point's proof sink (`NullSink`
/// under Competition, the streaming proof-hint bundle under Proof).
pub(crate) mod proof_bundle;
pub(crate) mod proof_check;
pub(crate) mod relational_equiv;
pub(crate) mod safenlp_input_head;
pub(crate) mod terminal_peel;
#[cfg(test)]
mod tests;
pub(crate) mod tll_structure;
pub(crate) mod tutorial;
/// #soundness-spine M2b: the only door to a published `unsat`.
pub(crate) mod unsat_receipt;
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

/// Isolate ORT shape inference in a child so a native crash can fall back to
/// exact static inference. Unit tests use the in-process backend because their
/// executable is the libtest harness, not the hidden shape-inference server;
/// integration tests exercise the real CLI subprocess path.
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

/// Find either supported VNN-COMP 2026 checkout layout. Callers fail explicitly
/// when neither directory exists.
#[cfg(all(test, feature = "mip", feature = "external-vnncomp"))]
pub(crate) fn vnncomp2026_benchmarks_root() -> Option<std::path::PathBuf> {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        manifest.join("../../benchmarks/vnncomp2026_benchmarks/benchmarks"),
        manifest.join("../../benchmarks/vnncomp2026/benchmarks"),
    ]
    .into_iter()
    .find(|candidate| candidate.is_dir())
}
