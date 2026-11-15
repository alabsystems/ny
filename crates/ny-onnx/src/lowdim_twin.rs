// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-lane — the LANE-PRIVATE interval twin of a model, for the
//! `#lowdim-prover` lane only.
//!
//! # Why
//!
//! The exact loader folds a both-constant node only when the result is EXACT
//! in binary32. `lsnc_relu`'s Lyapunov certificate has four constant-only
//! `MatMul` cones (the matrix `P = eps I + W^T W`, `0 @ P`, and the two
//! equilibrium-point network evaluations) whose rounded products are not the
//! authored products, so they fail closed to `OpaqueSkip` and every bound on
//! the real graph is `[-inf, +inf]`. The loader's interval fallback
//! (`const_fold::interval_fold`, the `NY_CONST_MATMUL_INTERVAL` carrier)
//! computes an outward-rounded ENCLOSURE of exactly those cones — the exact
//! real value over the stored binary32 leaves AND any binary32 implementation —
//! but that carrier is `Deliver::EnvOnly` and dark by policy: globally it would
//! hand the enclosure to EVERY bound path on a family that has published false
//! proofs twice.
//!
//! # What this does (owner decision, 2026-09-24)
//!
//! [`build_lane_twin`] loads a SECOND copy of the model with the same loader
//! configuration, under a thread-scoped guard that enables the interval
//! fallback for that one load, and moves the graph straight into
//! [`ny_propagate::lowdim::LaneTwin`]. That type keeps the graph in a private
//! field that only the lowdim lane reads; the lane's lowering carries every
//! enclosure WITH its radius. The real graph (the one every other lane bounds
//! on), the global lever, and the typed-preset refusal are all unchanged.
//!
//! The caller attaches the twin to the real graph
//! (`GraphNetwork::attach_lowdim_twin`) only when the real graph is the
//! unmodified model the twin was loaded from (no terminal peel, no load-time
//! rewrite); the binding is invalidated by any later structural mutation.

use std::path::Path;

use ny_propagate::lowdim::LaneTwin;

use crate::{load_onnx_with_config, GraphNetworkOptions, OnnxLoadConfig};

/// Load the lane-private interval twin of the model at `model_path`.
///
/// `load_config` and `options` must be the ones the real graph was built with,
/// so the twin differs from it only in the constant cones the exact folder
/// declined. Single-input models only (the lane's contract).
///
/// # Errors
///
/// A human-readable reason when the load fails, when no constant cone needed
/// an enclosure (the twin would equal the real graph), when the model has more
/// than one declared input, or when the twin graph still carries an
/// `OpaqueSkip` (see [`LaneTwin::new`]). The caller treats every error as "no
/// twin", which leaves today's behaviour unchanged.
pub fn build_lane_twin(
    model_path: &Path,
    load_config: &OnnxLoadConfig,
    options: GraphNetworkOptions,
) -> Result<LaneTwin, String> {
    let model = {
        let _scope = crate::loader::LanePrivateIntervalScope::enter();
        load_onnx_with_config(model_path, load_config)
            .map_err(|error| format!("lane-private load failed: {error}"))?
    };
    let interval_weights = model.weights.interval_len();
    if interval_weights == 0 {
        return Err(
            "no constant cone needed an enclosure; the twin would equal the real graph".into(),
        );
    }
    if model.network.inputs.len() != 1 {
        return Err(format!(
            "{} declared inputs (the lane twin is single-input only)",
            model.network.inputs.len()
        ));
    }
    let graph = model
        .to_graph_network_with_options(options)
        .map_err(|error| format!("lane-private graph build failed: {error}"))?;
    LaneTwin::new(graph, interval_weights)
}
