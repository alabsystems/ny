// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#[doc(hidden)]` debug/test exports for the GPU per-domain β machinery
//! (#w4-split-tightening).
//!
//! The β-gradient PARITY and bound-MONOTONICITY tests need a real GPU engine
//! (`ny-gpu` depends on `ny-propagate`, so they live in `ny-gpu`) while the CPU
//! analytic β optimizer and the GPU lane internals are crate-private here.
//! These wrappers expose exactly the two measurement points on a primitives-only
//! surface:
//!
//! * [`debug_cpu_beta_a_at_relu`] — the CPU capture (`a_at_relu`, the A-matrix
//!   at each split ReLU's OUTPUT before the ReLU relaxation) that feeds
//!   `GraphBetaState::compute_gradients_for_spec_row`
//!   (`∂lb_row/∂β_k = −sign_k · A[row, k]`).
//! * [`debug_gpu_beta_gather`] — the GPU gather of the same A-values from the
//!   sound resident resnet backward (`crown_backward_gpu_resnet_sound_beta_grad`).
//! * [`debug_gpu_beta_opt_vs_single`] — the production per-domain β ascent
//!   (`gpu_beta_optimize_domain`) vs the single-shot inherited-β pass, for the
//!   never-looser assertion.
//!
//! NOT a stable API; production code must not call these.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Array2;
use ny_core::{GemmEngine, GpuCrownSeed, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::domain::GraphCrownContext;
use crate::beta_crown::state::{GraphBetaEntry, GraphBetaState};
use crate::GraphNetwork;

use super::super::BetaCrownVerifier;

/// One ReLU split: `(node_name, neuron_idx, is_active, beta_value)`.
/// `is_active` ⇒ constraint `x ≥ 0`, sign `+1`; else `x ≤ 0`, sign `−1`.
pub type DebugSplit = (String, usize, bool, f32);

/// GPU-side gather output for the parity test.
pub struct GpuBetaDebugOut {
    /// ReLU node names in GPU fold order.
    pub relu_names: Vec<String>,
    /// Per-ReLU gathered neuron columns (fold order, aligned with `relu_names`).
    pub gather_idx: Vec<Vec<u32>>,
    /// Per-ReLU gathered lower A-values, row-major `num_specs × gather_idx[r].len()`.
    pub gathers: Vec<Vec<f32>>,
    /// Sound β-folded lower bounds per spec row.
    pub lower: Vec<f32>,
    /// Sound β-folded upper bounds per spec row.
    pub upper: Vec<f32>,
}

fn build_history_and_beta(splits: &[DebugSplit]) -> Result<(GraphSplitHistory, GraphBetaState)> {
    let mut history = GraphSplitHistory::new();
    let mut entries = Vec::with_capacity(splits.len());
    for (node, idx, is_active, value) in splits {
        history.add_constraint(GraphNeuronConstraint::new(
            node.clone(),
            *idx,
            *is_active,
            0.0,
        )?);
        let sign = if *is_active { 1.0 } else { -1.0 };
        entries.push(GraphBetaEntry::new(node.clone(), *idx, 0.0, *value, sign)?);
    }
    Ok((history, GraphBetaState::from_entries(entries)))
}

/// CPU capture point: run the constrained spec-matrix backward with the split
/// β folded and return `a_at_relu` (node → `num_specs × nn` lower A at that
/// ReLU's output) plus the per-row `(lower, upper)` spec bounds.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn debug_cpu_beta_a_at_relu(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    splits: &[DebugSplit],
) -> Result<(HashMap<String, Array2<f32>>, Vec<(f32, f32)>)> {
    let (history, beta) = build_history_and_beta(splits)?;
    let verifier = BetaCrownVerifier::default();
    let (bounds_cache, constrained_input) =
        verifier.compute_constrained_forward_bounds(graph, input, &history, None, None)?;
    // #cone-delta increment 2: the forward cache is already `Arc`-shared.
    let context = GraphCrownContext::new(&history, None, Some(&bounds_cache), None);
    let (output, _node_bounds, intermediate) = verifier
        .propagate_crown_with_graph_beta_and_spec_matrix_storing_intermediates(
            graph,
            &constrained_input,
            &context,
            &beta,
            spec_matrix,
        )?;
    let flat = output.flatten();
    let rows = (0..flat.len())
        .map(|i| (flat.lower()[[i]], flat.upper()[[i]]))
        .collect();
    Ok((intermediate.a_at_relu, rows))
}

/// Shared GPU setup: constrained forward bounds + resnet segments + the β fold
/// / gather inputs, mirroring `try_gpu_beta_batched_resnet` exactly.
#[allow(clippy::type_complexity)]
fn gpu_setup(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    beta: &GraphBetaState,
    history: &GraphSplitHistory,
) -> Option<(
    Vec<ny_core::GpuResnetSegment>,
    Vec<String>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<u32>>,
    GpuCrownSeed,
    Vec<f32>,
    Vec<f32>,
    HashMap<String, Arc<BoundedTensor>>,
)> {
    let verifier = BetaCrownVerifier::default();
    let (bounds_cache, constrained_input) = verifier
        .compute_constrained_forward_bounds(graph, input, history, None, None)
        .ok()?;
    let (segments, relu_names, frontier_abs, node_abs) =
        crate::network::extract_gpu_resnet_segments_with_relu_names(
            graph,
            &constrained_input,
            &graph.output_node,
            &bounds_cache,
            &bounds_cache,
            None,
        )?;
    let mut beta_signed: Vec<Vec<f32>> = Vec::with_capacity(relu_names.len());
    let mut gather_idx: Vec<Vec<u32>> = Vec::with_capacity(relu_names.len());
    for name in &relu_names {
        let nn = bounds_cache.get(name)?.lower().len();
        let mut bs = vec![0.0f32; nn];
        let mut gi: Vec<u32> = Vec::new();
        for e in beta.entries_for_node(name) {
            if e.split_point().abs() < 1e-6 && e.neuron_idx() < nn {
                bs[e.neuron_idx()] = e.signed_value();
                let col = e.neuron_idx() as u32;
                if !gi.contains(&col) {
                    gi.push(col);
                }
            }
        }
        beta_signed.push(bs);
        gather_idx.push(gi);
    }
    let num_specs = spec_matrix.nrows();
    let output_dim = spec_matrix.ncols();
    let seed_rows: Vec<f32> = spec_matrix.iter().copied().collect();
    let seed = GpuCrownSeed {
        lower_a: seed_rows.clone().into(),
        upper_a: seed_rows.into(),
        lower_b: vec![0.0f32; num_specs].into(),
        upper_b: vec![0.0f32; num_specs].into(),
        num_specs,
        current_dim: output_dim,
    };
    let in_lo: Vec<f32> = constrained_input.lower().iter().copied().collect();
    let in_hi: Vec<f32> = constrained_input.upper().iter().copied().collect();
    Some((
        segments,
        relu_names,
        frontier_abs,
        node_abs,
        beta_signed,
        gather_idx,
        seed,
        in_lo,
        in_hi,
        bounds_cache,
    ))
}

/// GPU capture point: one sound β-folded resnet backward with the A-value
/// gather at the split neurons (the production β-gradient input).
#[doc(hidden)]
pub fn debug_gpu_beta_gather(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    splits: &[DebugSplit],
    engine: &dyn GemmEngine,
) -> Option<GpuBetaDebugOut> {
    let (history, beta) = build_history_and_beta(splits).ok()?;
    let gpu = engine
        .as_gpu_crown_backward()
        .filter(|g| g.provides_sound_gpu_crown())?;
    let (
        segments,
        relu_names,
        frontier_abs,
        node_abs,
        beta_signed,
        gather_idx,
        seed,
        in_lo,
        in_hi,
        _cache,
    ) = gpu_setup(graph, input, spec_matrix, &beta, &history)?;
    let result = gpu
        .crown_backward_gpu_resnet_sound_beta_grad(
            &segments,
            &seed,
            &in_lo,
            &in_hi,
            &beta_signed,
            &gather_idx,
            &frontier_abs,
            &node_abs,
        )
        .ok()?;
    Some(GpuBetaDebugOut {
        relu_names,
        gather_idx,
        gathers: result.beta_gather,
        lower: result.lower_bounds,
        upper: result.upper_bounds,
    })
}

/// #interm-refine oracle surface: build ONE BaB subdomain from `(input,
/// splits)` — the constrained forward bounds land the split clamps on the seed
/// node's cache entry, exactly as the production forward does — then run the
/// production per-subdomain intermediate refinement
/// (`refine_last_relu_interm_bounds`) and return
/// `(seed_node, inherited (l,u), refined (l,u))` for the last ReLU's
/// pre-activations. `None` mirrors production refusal (no sound GPU, no clean
/// last-ReLU chain, refinement declined).
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn debug_interm_refine_last_relu(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    splits: &[DebugSplit],
    engine: &dyn GemmEngine,
) -> Option<(String, Vec<(f32, f32)>, Vec<(f32, f32)>)> {
    let (history, beta) = build_history_and_beta(splits).ok()?;
    let verifier = BetaCrownVerifier::default();
    let (bounds_cache, constrained_input) = verifier
        .compute_constrained_forward_bounds(graph, input, &history, None, None)
        .ok()?;
    let (_relu_name, seed_node) =
        super::propagation::batched::interm_refine::find_last_relu_seed(graph, &graph.output_node)?;
    let pairs = |bt: &BoundedTensor| -> Vec<(f32, f32)> {
        bt.lower()
            .iter()
            .zip(bt.upper().iter())
            .map(|(&l, &u)| (l, u))
            .collect()
    };
    let inherited = pairs(bounds_cache.get(&seed_node)?);
    let caches = [bounds_cache];
    let inputs = [constrained_input];
    let betas: [Option<&GraphBetaState>; 1] = [Some(&beta)];
    let alphas: [Option<&crate::beta_crown::state::GraphDomainAlphaState>; 1] = [None];
    // Empty spec matrix: this debug harness does not exercise the joint
    // margin-α lane (dim-mismatch ⇒ uniform fallback in `compute_margin_weights`).
    let spec_matrix = Array2::<f32>::zeros((0, 0));
    let outcome = verifier.refine_last_relu_interm_bounds(
        graph,
        &graph.output_node,
        1,
        &caches,
        &inputs,
        &betas,
        &alphas,
        engine,
        &spec_matrix,
    )?;
    let refined = pairs(outcome.caches.first()?.get(&seed_node)?);
    Some((seed_node, inherited, refined))
}

/// Never-looser harness: `(single_shot, optimized)` per-row bounds from the
/// SAME inherited-β state — `single_shot` is the legacy GPU lane call,
/// `optimized` runs the production per-domain analytic β ascent
/// (`gpu_beta_optimize_domain`, `iterations` steps, all rows unverified,
/// thresholds as given).
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn debug_gpu_beta_opt_vs_single(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    splits: &[DebugSplit],
    thresholds: &[f32],
    iterations: usize,
    engine: &dyn GemmEngine,
) -> Option<((Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>))> {
    let (history, beta) = build_history_and_beta(splits).ok()?;
    let mut verifier = BetaCrownVerifier::default();
    verifier.config.beta_iterations = iterations;
    let gpu = engine
        .as_gpu_crown_backward()
        .filter(|g| g.provides_sound_gpu_crown())?;
    let (
        segments,
        relu_names,
        frontier_abs,
        node_abs,
        beta_signed,
        _gather_idx,
        seed,
        in_lo,
        in_hi,
        cache,
    ) = gpu_setup(graph, input, spec_matrix, &beta, &history)?;
    let num_specs = spec_matrix.nrows();
    let single = gpu
        .crown_backward_gpu_resnet_sound_beta(
            &segments,
            &seed,
            &in_lo,
            &in_hi,
            &beta_signed,
            &frontier_abs,
            &node_abs,
        )
        .ok()?;
    let row_verified = vec![false; num_specs];
    let (opt_lo, opt_hi, _best_beta) = verifier.gpu_beta_optimize_domain(
        gpu,
        &segments,
        &relu_names,
        &frontier_abs,
        &node_abs,
        &seed,
        &in_lo,
        &in_hi,
        &cache,
        &beta,
        beta_signed,
        thresholds,
        &row_verified,
        num_specs,
    )?;
    Some(((single.lower_bounds, single.upper_bounds), (opt_lo, opt_hi)))
}
