// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dense-spec batched bounds helper for multi-objective input-split paths.
//!
//! Extracted from `shared.rs` to stay under the 500-line file limit.
//! Part of #4116 Packet A Step 4.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, NaiveCpuGemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::batched_domain::{BatchedDomainOptions, BatchedDomainsBuilder};
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::graph::propagation::BatchedBackwardContext;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::state::GraphDomainAlphaState;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::faer_parallelism::RayonTaskGuard;
use crate::GraphNetwork;

use super::metrics::{DenseSpecReboundMode, DenseSpecReboundTiming};
use super::shared::compute_crown_or_ibp_bounds;

/// Dense-spec batched bounds: preserves full BoundedTensor and LinearBounds per domain.
///
/// Unlike `BatchedScalarBounds` which collapses to scalar lower/upper pairs, this
/// carrier preserves the multi-row BoundedTensor output and the full input LinearBounds
/// for each domain. Used by the dense-spec input-split helper.
///
/// Part of #4116 Packet A Step 4.
#[derive(Debug)]
pub(crate) struct BatchedSpecBounds {
    pub(crate) bounds: Vec<BoundedTensor>,
    pub(crate) linear_bounds: Vec<Option<LinearBounds>>,
    pub(crate) rebound_timing: DenseSpecReboundTiming,
}

/// Compute CROWN/IBP bounds for a batch of input domains in parallel (dense-spec).
///
/// Accepts any spec matrix (single-row or multi-row) and returns full `BoundedTensor`
/// output plus input `LinearBounds` per domain. This is the true dense-spec carrier
/// that later packets can use for multi-objective input-split batching.
///
/// Uses the graph batched-backward kernel when the input-split feature set matches
/// that surface. Supports α-CROWN global alpha state (#4210) and shared reference
/// bounds. Falls back to rayon-parallel per-domain CROWN for MulBinary alpha cache,
/// truncated CROWN backward, or IBP-enhanced reference-bound merge.
///
/// Part of #4116 Packet A Step 4.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_crown_or_ibp_bounds_batched_specs(
    graph: &GraphNetwork,
    input_bounds_batch: &[&BoundedTensor],
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    ibp_enhancement: bool,
    stacked_rebound: bool,
) -> Result<BatchedSpecBounds> {
    if let Some(result) = try_compute_crown_or_ibp_bounds_batched_specs_with_batched_backward(
        graph,
        input_bounds_batch,
        spec_matrix,
        engine,
        alpha_node_bounds,
        alpha_state,
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
        ibp_enhancement,
        stacked_rebound,
    )? {
        return Ok(result);
    }

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    info!(
        domains = input_bounds_batch.len(),
        mul_binary_alphas = mul_binary_alphas.is_some(),
        crown_backward_layers = crown_backward_layers.is_some(),
        "input_split: batched kernel gated, falling back to rayon par_iter"
    );

    let total_start = Instant::now();
    let results: Result<Vec<(BoundedTensor, Option<LinearBounds>)>> = input_bounds_batch
        .par_iter()
        .map(|input_bounds| {
            let _rayon_task_guard = RayonTaskGuard::new();
            compute_crown_or_ibp_bounds(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                alpha_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
                ibp_enhancement,
            )
        })
        .collect();

    let results = results?;
    let n = results.len();
    let mut bounds = Vec::with_capacity(n);
    let mut linear_bounds = Vec::with_capacity(n);

    for (bt, linear) in results {
        bounds.push(bt);
        linear_bounds.push(linear);
    }

    Ok(BatchedSpecBounds {
        bounds,
        linear_bounds,
        rebound_timing: DenseSpecReboundTiming {
            mode: DenseSpecReboundMode::RayonFallback,
            domains: input_bounds_batch.len(),
            num_specs: spec_matrix.nrows(),
            total_elapsed_s: total_start.elapsed().as_secs_f64(),
            forward_elapsed_s: None,
            backward_elapsed_s: None,
            materialize_elapsed_s: None,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn try_compute_crown_or_ibp_bounds_batched_specs_with_batched_backward(
    graph: &GraphNetwork,
    input_bounds_batch: &[&BoundedTensor],
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    ibp_enhancement: bool,
    stacked_rebound: bool,
) -> Result<Option<BatchedSpecBounds>> {
    // Gate: fall back to rayon par_iter for features the batched kernel
    // doesn't yet support.
    // Part of #4210: alpha_state gate lifted by converting
    //   GraphAlphaState → GraphDomainAlphaState.
    // Part of #4210: ibp_enhancement gate lifted — the batched kernel's
    //   `compute_constrained_forward_bounds` seeds from `base_bounds`
    //   (warmup alpha_node_bounds) which gives valid intermediate bounds.
    //   The per-domain ibp_enhancement path additionally intersects fresh
    //   subdomain IBP with the warmup bounds; the batched path omits this
    //   extra IBP pass UNLESS `stacked_rebound` is also enabled
    //   (#cgan-batched-stack: `input_split_batched_ibp_refresh` restores the
    //   per-domain intersect so relaxations re-anchor as boxes shrink).
    // Part of #4284: mul_binary_alphas gate lifted — root-optimized alphas
    //   are shared across all domains and threaded through the batched
    //   backward context to DispatchContext per-node.
    if input_bounds_batch.is_empty() || crown_backward_layers.is_some() {
        return Ok(None);
    }

    // Convert alpha_node_bounds (HashMap<String, BoundedTensor>) to Arc-wrapped
    // form (HashMap<String, Arc<BoundedTensor>>) for the batched backward context.
    // In input-split, all sub-domains share the same warmup reference bounds from
    // the initial α-CROWN pass ("fix_interm_bounds" pattern).
    // Ref: alpha-beta-CROWN input_split/branching_domains.py:
    //   reference bounds are shared, not per-domain.
    let arc_node_bounds: Option<HashMap<String, Arc<BoundedTensor>>> =
        alpha_node_bounds.map(|bounds| {
            bounds
                .iter()
                .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
                .collect()
        });

    // Convert global α-CROWN state to per-neuron format for the batched kernel.
    // All input-split sub-domains share the same global alpha from the warmup
    // pass ("fix_interm_bounds" pattern). The converter stores entries only for
    // neurons that were unstable at the root level; the batched backward core's
    // build_alpha_array handles per-domain stable/unstable classification at
    // runtime using per-domain pre-activation bounds. Part of #4210.
    let domain_alpha: Option<GraphDomainAlphaState> =
        alpha_state.map(GraphDomainAlphaState::from_global_alpha_state_for_input_split);

    let mut builder =
        BatchedDomainsBuilder::new_with_options(Vec::new(), BatchedDomainOptions::default());
    let empty_layer_bounds: HashMap<String, (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> =
        HashMap::new();
    for input_bounds in input_bounds_batch {
        builder.add_domain(
            &empty_layer_bounds,
            input_bounds.lower().clone(),
            input_bounds.upper().clone(),
            0.0,
            0.0,
            0,
            Vec::new(),
        );
    }
    let batched = builder.build()?;

    let empty_history = GraphSplitHistory::new();
    let histories = vec![&empty_history; input_bounds_batch.len()];
    let n = input_bounds_batch.len();

    let ctx = BatchedBackwardContext {
        batched: &batched,
        histories,
        beta_states: vec![None; n],
        base_bounds: match &arc_node_bounds {
            Some(bounds) => vec![Some(bounds); n],
            None => vec![None; n],
        },
        // #cone-delta: shared warmup maps have no tracked delta — fail closed.
        delta_seeds: vec![None; n],
        alpha_states: match &domain_alpha {
            Some(state) => vec![Some(state); n],
            None => vec![None; n],
        },
        cached_la: vec![None; n],
        mul_binary_alphas, // #4284: thread root-optimized MulBinary alphas
    };

    let cpu_engine = NaiveCpuGemmEngine;
    // #cgan-batched-stack: under the stacked-rebound gate, the CPU fallback
    // engine is the faer-backed blocked SIMD GEMM instead of the naive scalar
    // triple loop — the stacked conv backward GEMMs (25088x512x64-class rows
    // on cgan) run at memory bandwidth instead of ~2 GFLOP/s. Same RN-f32
    // precision contract (any summation order; certified error bounds are
    // order-independent). Flag off => byte-identical historical engine.
    let faer_engine = crate::faer_parallelism::FaerCpuGemmEngine;
    // #linearizenn-batched-faer: when the caller supplies no GEMM engine (every
    // CPU `ny vnncomp` / `ny beta-crown` run — `compute_device` is None for the
    // Cpu backend), the batched dense-spec rebound's per-domain CROWN backward
    // GEMMs default to faer's blocked SIMD matmul instead of the scalar
    // triple-loop `NaiveCpuGemmEngine`. On linearizenn AllInOne_120_120 the
    // input-split BaB re-bounds ~36 domains/s with the naive engine — a
    // sampling profile showed `NaiveCpuGemmEngine::gemm_f32` at ~42% of all CPU,
    // the per-domain 120-wide CROWN backward being the single hottest leaf. faer
    // ~doubles throughput (measured 3250→6322 bounded domains in a fixed 90s
    // budget, ~36→~70 domains/s, 1.95x) with parallelism already near-saturated
    // (~12.3/14 cores). This is a pure sound throughput win; it does NOT by
    // itself flip AllInOne_120_120 prop_120_120_0 (the proof tree is >60k
    // domains, so ~2x is still ~1.5x short of the 855s budget — after faer the
    // remaining ~75% of CPU is the sound f64 certified backward
    // `aw_f64_with_abssum`, which is off-limits to a throughput-only change).
    //
    // SOUNDNESS: this changes ONLY the GEMM engine, never the bound math. Both
    // engines honor the same `GemmEngine::gemm_f32` contract — plain RN-f32
    // arithmetic in ANY summation order — and the verdict-feeding CROWN callers'
    // certified coefficient-error term (γ_n·S, the `aw_f64_with_abssum` path) is
    // summation-order independent (the identical Higham argument that lets these
    // products route through cuBLAS/faer elsewhere, e.g. the stacked-rebound
    // lever above). A per-domain bound may differ from the naive path by a few
    // ULP but is equally sound; a tighter/looser sound bound only ever changes
    // how many domains verify, never the verdict. Inside the rayon per-domain
    // workers `current_par()` forces faer to `Par::Seq`, so there is no nested
    // Rayon (#4392). Set `NY_BATCHED_NAIVE_ENGINE=1` to restore the historical
    // byte-identical naive engine (the gate-OFF A/B + parity reference).
    let force_naive = std::env::var_os("NY_BATCHED_NAIVE_ENGINE").is_some();
    let resolved_engine: &dyn GemmEngine = match engine {
        Some(engine) => engine,
        None if stacked_rebound => &faer_engine,
        None if !force_naive => &faer_engine,
        None => &cpu_engine,
    };
    let mut verifier_config = BetaCrownConfig::default();
    verifier_config.alpha_config.deadline = deadline;
    // #cgan-batched-stack: enable the domain-stacked conv/BN backward, and the
    // fresh per-domain IBP intersect when the preset also asks for
    // ibp_enhancement. Both default off (byte-identical historical kernel).
    verifier_config.input_split_stacked_rebound = stacked_rebound;
    verifier_config.input_split_batched_ibp_refresh = stacked_rebound && ibp_enhancement;
    let mut verifier = BetaCrownVerifier::new(verifier_config);
    // This is a nested propagation adapter, so the caller's deadline is the
    // complete authority for this rebound. `BetaCrownVerifier::new` normally
    // anchors a fresh top-level timeout; restore the explicitly supplied value
    // (including `None`) so unscored callers retain the historical batched GEMM
    // path while finite-deadline callers remain on the pollable path.
    //
    // #cgan-row7 DIAGNOSTIC (2026-08-12). This assignment is the one change in
    // `6f49a660` — the bisected first-bad commit for the cgan row-7
    // `unsat -> timeout` regression — that lands in the input-split BaB family
    // row 7 actually runs. A FINITE deadline here is a throughput cliff, not a
    // neutral hand-off: `Conv2dLayer::propagate_ibp_with_engine_and_deadline`
    // documents that with one set, "neither the caller engine nor faer's
    // unpollable GEMM is entered" and dense convs fall back to a direct scalar
    // CPU contraction. Row 7 spends ~584 s in this phase.
    //
    // `NY_INPUT_SPLIT_NESTED_DEADLINE=0` restores the pre-`6f49a660` shape for
    // the A/B ONLY. It is diagnostic: dropping the deadline removes this
    // rebound's interruptibility, so it must never be the shipped default.
    // Declared as an OPT-OUT: the shipped arm is `true` (keep the deadline), and
    // only an exact "0" disarms it, so `!as_bool()` is the drop. The chokepoint
    // resolves any other present value back to the shipped default, which is the
    // safe direction for a lever whose off-arm removes interruptibility.
    use ny_levers::decls::dark_probes::{INPUT_SPLIT_NESTED_DEADLINE, INPUT_SPLIT_PROBE};
    let drop_nested_deadline = !ny_levers::read(&INPUT_SPLIT_NESTED_DEADLINE)
        .value
        .as_bool();
    verifier.config.alpha_config.deadline = if drop_nested_deadline { None } else { deadline };
    if ny_levers::read(&INPUT_SPLIT_PROBE).value.as_bool() {
        eprintln!(
            "[input-split-rebound] domains={n} deadline_finite={} dropped={drop_nested_deadline} \
             stacked_rebound={stacked_rebound}",
            deadline.is_some()
        );
    }

    info!(
        domains = n,
        shared_root_bounds = arc_node_bounds.is_some(),
        shared_alpha = domain_alpha.is_some(),
        mul_binary_alphas = mul_binary_alphas.is_some(),
        stacked_rebound = stacked_rebound,
        ibp_refresh = stacked_rebound && ibp_enhancement,
        "input_split: using batched backward fast-path"
    );

    let total_start = Instant::now();
    let results = verifier.propagate_crown_batched_with_context_specs_timed(
        graph,
        &ctx,
        spec_matrix,
        resolved_engine,
    )?;

    let stage_timing = results.stage_timing;
    let batched_results = results.results;
    let mut bounds = Vec::with_capacity(batched_results.len());
    let mut linear_bounds = Vec::with_capacity(batched_results.len());
    for result in batched_results {
        bounds.push(result.output_bounds);
        linear_bounds.push(result.input_linear);
    }

    Ok(Some(BatchedSpecBounds {
        bounds,
        linear_bounds,
        rebound_timing: DenseSpecReboundTiming {
            mode: DenseSpecReboundMode::BatchedFastPath,
            domains: input_bounds_batch.len(),
            num_specs: spec_matrix.nrows(),
            total_elapsed_s: total_start.elapsed().as_secs_f64(),
            forward_elapsed_s: stage_timing.map(|timing| timing.forward_elapsed_s),
            backward_elapsed_s: stage_timing.map(|timing| timing.backward_elapsed_s),
            materialize_elapsed_s: stage_timing.map(|timing| timing.materialize_elapsed_s),
        },
    }))
}
