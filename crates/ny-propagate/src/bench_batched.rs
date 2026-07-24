// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Public benchmarking shim for the batched dense-spec CROWN backward.
//!
//! The M2 (batched/shared-root disjunctive BaB) throughput gate needs to time
//! the SAME batch-safe primitive that the shared-root lane uses —
//! `compute_crown_or_ibp_bounds_batched_specs` — over a batch of sub-boxes of a
//! real ONNX net (mscn_2048d_dual). That primitive and its `BatchedDomains`
//! plumbing are crate-internal (`pub(crate)`), and ONNX loading lives one crate
//! up (ny-onnx depends on ny-propagate), so the CLI cannot reach the primitive
//! directly. This thin, genuinely-public wrapper bridges the two: the CLI (or an
//! example) hands over a `GraphNetwork`, a batch of input boxes, and a spec
//! matrix; we run one shared-root batched dense-spec CROWN backward and return
//! the stage timing.
//!
//! This is a measurement-only entry — it computes SOUND bounds (identical math
//! to the input-split rebound) but is never on a verdict path.

use std::collections::HashMap;

use ndarray::Array2;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::engine::graph::input_split::shared_specs::compute_crown_or_ibp_bounds_batched_specs;
use crate::{DenseSpecReboundMode, GraphNetwork};

/// Timing + sanity output of one batched dense-spec CROWN backward pass.
#[derive(Debug, Clone)]
pub struct BatchedBackwardBench {
    /// Number of sub-box domains (leaves) processed in the one batched call.
    pub n_domains: usize,
    /// Number of spec rows (output directions) propagated together.
    pub num_specs: usize,
    /// True when the true batched fast-path kernel ran; false when the call
    /// gated out to the rayon per-domain fallback (a materially different,
    /// per-domain throughput regime the caller must report distinctly).
    pub batched_fast_path: bool,
    /// Wall time of the whole rebound call (forward + backward + materialize).
    pub total_elapsed_s: f64,
    /// Forward-pass wall time, when the batched kernel retained stage timing.
    pub forward_elapsed_s: Option<f64>,
    /// Backward-pass wall time, when the batched kernel retained stage timing.
    pub backward_elapsed_s: Option<f64>,
    /// First domain, first spec row: concretized lower bound (sanity check).
    pub sample_row0_lb: f32,
    /// First domain, first spec row: concretized upper bound (sanity check).
    pub sample_row0_ub: f32,
}

/// Run ONE shared-root batched dense-spec CROWN backward over `input_boxes`
/// with the given `spec_matrix`, returning stage timing. `shared_root_bounds`
/// (from e.g. `GraphNetwork::collect_forward_linear_bounds_dag_with_engine` on
/// the parent box) supplies the reference intermediate bounds shared across all
/// leaves — the shared-root regime M2 relies on; pass `None` to measure the
/// self-contained per-leaf forward+backward cost.
pub fn bench_batched_dense_spec_backward(
    graph: &GraphNetwork,
    input_boxes: &[BoundedTensor],
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    shared_root_bounds: Option<&HashMap<String, BoundedTensor>>,
) -> Result<BatchedBackwardBench> {
    let refs: Vec<&BoundedTensor> = input_boxes.iter().collect();
    let out = compute_crown_or_ibp_bounds_batched_specs(
        graph,
        &refs,
        spec_matrix,
        engine,
        shared_root_bounds,
        None,  // alpha_state
        None,  // mul_binary_alphas
        None,  // deadline
        None,  // crown_backward_layers (None => batched fast-path eligible)
        false, // ibp_enhancement
        false, // stacked_rebound
    )?;

    let timing = &out.rebound_timing;
    let (lb, ub) = out
        .bounds
        .first()
        .and_then(|bt| {
            let lo = bt.lower();
            let hi = bt.upper();
            match (lo.as_slice(), hi.as_slice()) {
                (Some(l), Some(h)) if !l.is_empty() && !h.is_empty() => Some((l[0], h[0])),
                _ => None,
            }
        })
        .unwrap_or((f32::NAN, f32::NAN));

    Ok(BatchedBackwardBench {
        n_domains: timing.domains,
        num_specs: timing.num_specs,
        batched_fast_path: matches!(timing.mode, DenseSpecReboundMode::BatchedFastPath),
        total_elapsed_s: timing.total_elapsed_s,
        forward_elapsed_s: timing.forward_elapsed_s,
        backward_elapsed_s: timing.backward_elapsed_s,
        sample_row0_lb: lb,
        sample_row0_ub: ub,
    })
}
