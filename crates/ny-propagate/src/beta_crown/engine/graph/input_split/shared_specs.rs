// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dense-spec batched bounds helper for multi-objective input-split paths.
//!
//! Extracted from `shared.rs` to stay under the 500-line file limit.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, NaiveCpuGemmEngine, NyError, Result};
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
/// that surface. Supports α-CROWN global alpha state and shared reference
/// bounds. Falls back to rayon-parallel per-domain CROWN for MulBinary alpha cache,
/// truncated CROWN backward, or IBP-enhanced reference-bound merge.
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
    nested_deadline: Option<bool>,
) -> Result<BatchedSpecBounds> {
    // The batched backward kernel does not yet apply the post-CROWN joint
    // `ReduceSum(A * B)` intersection performed by the scalar evaluator in
    // `bounds_eval.rs`.  Letting an armed joint relaxation take this fast path
    // would silently bypass the lever on reordered dense-spec rebounds.  Treat
    // it as an unsupported batched feature until the kernel has a joint-capable
    // result stage; the existing per-domain fallback below is shrink-only,
    // sound, and already runs the joint hook.  The shipped OFF arm enters the
    // byte-identical historical fast-path call.
    let joint_bilinear_armed = crate::joint_bilinear::joint_bilinear_armed();
    let joint_bilinear_requires_scalar = joint_bilinear_armed
        && input_bounds_batch
            .iter()
            .any(|input| crate::joint_bilinear::joint_bilinear_may_apply(graph, input));
    if !joint_bilinear_requires_scalar {
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
            nested_deadline,
            None,
        )? {
            return Ok(result);
        }
    }

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    info!(
        domains = input_bounds_batch.len(),
        mul_binary_alphas = mul_binary_alphas.is_some(),
        crown_backward_layers = crown_backward_layers.is_some(),
        joint_bilinear_armed,
        joint_bilinear_requires_scalar,
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

/// `#cgan-child-zono`: [`compute_crown_or_ibp_bounds_batched_specs`] with a
/// PER-DOMAIN replacement for some entries of the shared reference map.
///
/// `domain_overrides[i]`, when present, holds node entries that REPLACE the
/// shared `alpha_node_bounds` entries of the same name for domain `i` only.
/// The caller (the armed zonotope prepass) guarantees every replacement is
/// the root entry intersected shrink-only with a certified enclosure of that
/// domain's box (`merge_reference_bound_maps`), so each per-domain map is a
/// valid enclosure of the domain's reachable set and no tighter than the
/// intersection of two such enclosures. Domains without replacements use the
/// shared map exactly as [`compute_crown_or_ibp_bounds_batched_specs`] does.
///
/// Only the armed prepass calls this; every other caller keeps the historical
/// shared-map entry point, whose call into the batched kernel is unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_crown_or_ibp_bounds_batched_specs_with_domain_overrides(
    graph: &GraphNetwork,
    input_bounds_batch: &[&BoundedTensor],
    domain_overrides: &[Option<HashMap<String, BoundedTensor>>],
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    ibp_enhancement: bool,
    stacked_rebound: bool,
    nested_deadline: Option<bool>,
) -> Result<BatchedSpecBounds> {
    if domain_overrides.len() != input_bounds_batch.len() {
        return Err(NyError::InvalidSpec(format!(
            "per-domain reference overrides: {} maps for {} domains",
            domain_overrides.len(),
            input_bounds_batch.len()
        )));
    }
    let joint_bilinear_requires_scalar = crate::joint_bilinear::joint_bilinear_armed()
        && input_bounds_batch
            .iter()
            .any(|input| crate::joint_bilinear::joint_bilinear_may_apply(graph, input));
    if !joint_bilinear_requires_scalar {
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
            nested_deadline,
            Some(domain_overrides),
        )? {
            return Ok(result);
        }
    }

    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    // Per-domain fallback (the kernel declined the batch): each domain gets
    // the FULL shared map with its replacements merged in, so the
    // no-enhancement route (which substitutes the child map for the shared
    // one) still sees every node.
    let total_start = Instant::now();
    let results: Result<Vec<(BoundedTensor, Option<LinearBounds>)>> = (0..input_bounds_batch.len())
        .into_par_iter()
        .map(|idx| {
            let _rayon_task_guard = RayonTaskGuard::new();
            let child_map = match domain_overrides[idx].as_ref() {
                Some(replacements) => crate::network::merge_reference_bound_maps(
                    alpha_node_bounds,
                    Some(replacements),
                )?,
                None => None,
            };
            super::shared::compute_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds_batch[idx],
                spec_matrix,
                engine,
                alpha_node_bounds,
                child_map.as_ref(),
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
                ibp_enhancement,
            )
        })
        .collect();
    let (bounds, linear_bounds): (Vec<BoundedTensor>, Vec<Option<LinearBounds>>) =
        results?.into_iter().unzip();
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
    nested_deadline: Option<bool>,
    domain_overrides: Option<&[Option<HashMap<String, BoundedTensor>>]>,
) -> Result<Option<BatchedSpecBounds>> {
    // Gate: fall back to rayon par_iter for features the batched kernel
    // doesn't yet support.
    // alpha_state gate lifted by converting
    //   GraphAlphaState → GraphDomainAlphaState.
    // ibp_enhancement gate lifted — the batched kernel's
    //   `compute_constrained_forward_bounds` seeds from `base_bounds`
    //   (warmup alpha_node_bounds) which gives valid intermediate bounds.
    //   The per-domain ibp_enhancement path additionally intersects fresh
    //   subdomain IBP with the warmup bounds; the batched path omits this
    //   extra IBP pass UNLESS `stacked_rebound` is also enabled
    //   (#cgan-batched-stack: `input_split_batched_ibp_refresh` restores the
    //   per-domain intersect so relaxations re-anchor as boxes shrink).
    // mul_binary_alphas gate lifted — root-optimized alphas
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
    // #cgan-child-zono: a domain with replacement entries gets the shared map
    // (the same `Arc`s, no tensor copies) with those entries swapped in. Only
    // the armed zonotope prepass passes `Some`; `None` leaves the shared map
    // below exactly as it was. Replacements are ignored without a shared map,
    // so a per-domain map is never missing a node the shared one carries.
    let per_domain_node_bounds: Option<Vec<Option<HashMap<String, Arc<BoundedTensor>>>>> =
        match (domain_overrides, &arc_node_bounds) {
            (Some(overrides), Some(shared)) => {
                if overrides.len() != input_bounds_batch.len() {
                    return Err(NyError::InvalidSpec(format!(
                        "per-domain reference overrides: {} maps for {} domains",
                        overrides.len(),
                        input_bounds_batch.len()
                    )));
                }
                Some(
                    overrides
                        .iter()
                        .map(|replacements| {
                            replacements.as_ref().map(|replacements| {
                                let mut map = shared.clone();
                                for (name, bounds) in replacements {
                                    map.insert(name.clone(), Arc::new(bounds.clone()));
                                }
                                map
                            })
                        })
                        .collect(),
                )
            }
            _ => None,
        };

    // Convert global α-CROWN state to per-neuron format for the batched kernel.
    // All input-split sub-domains share the same global alpha from the warmup
    // pass ("fix_interm_bounds" pattern). The converter stores entries only for
    // neurons that were unstable at the root level; the batched backward core's
    // build_alpha_array handles per-domain stable/unstable classification at
    // runtime using per-domain pre-activation bounds.
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
        base_bounds: match (&per_domain_node_bounds, &arc_node_bounds) {
            (Some(per_domain), shared) => per_domain
                .iter()
                .map(|map| map.as_ref().or(shared.as_ref()))
                .collect(),
            (None, Some(bounds)) => vec![Some(bounds); n],
            (None, None) => vec![None; n],
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
    // Rayon. Set `NY_BATCHED_NAIVE_ENGINE=1` to restore the historical
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
    // `NY_INPUT_SPLIT_NESTED_DEADLINE=0` restores the pre-`6f49a660` shape.
    // Historically diagnostic-only; #dist-shift-5291 adds the TYPED preset
    // layer `bab.branching.input_split.nested_deadline` (threaded here as
    // `nested_deadline`), so a category whose per-batch rebound cost is
    // known-small can ship the relaxation, with wall-clock safety provided by
    // the OUTER BaB loop's own deadline poll between batches plus the harness
    // per-instance budget. Layering (ny_levers::read_over_config): env
    // overrides in BOTH directions (`0` drop / `1` keep), absent env defers
    // to the preset field, absent both keeps the declaration default (keep).
    // A present-but-malformed env token resolves to the declaration default
    // (keep) and SUPPRESSES the preset — the safe direction for a lever whose
    // off-arm removes interruptibility.
    use ny_levers::decls::dark_probes::INPUT_SPLIT_PROBE;
    let (drop_nested_deadline, nested_deadline_source) = nested_deadline_dropped(nested_deadline);
    verifier.config.alpha_config.deadline = if drop_nested_deadline { None } else { deadline };
    if drop_nested_deadline {
        // R9 engagement fingerprint: once per process, unconditional, so any
        // scored log carries the lever's own observable move (and its source).
        static ANNOUNCED: std::sync::Once = std::sync::Once::new();
        ANNOUNCED.call_once(|| {
            eprintln!(
                "[input-split-rebound] nested deadline DROPPED (source: {nested_deadline_source}): \
                 rebound bounds run under the OUTER BaB deadline only"
            );
        });
    }
    if ny_levers::read(&INPUT_SPLIT_PROBE).value.as_bool() {
        eprintln!(
            "[input-split-rebound] domains={n} deadline_finite={} dropped={drop_nested_deadline} \
             source={nested_deadline_source} stacked_rebound={stacked_rebound}",
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

/// Resolve the nested-rebound deadline authority (#dist-shift-5291).
///
/// Layering is `ny_levers::read_over_config`: `NY_INPUT_SPLIT_NESTED_DEADLINE`
/// first (`0` = drop, `1` = keep), then the typed preset field
/// (`BetaCrownConfig::input_split_nested_deadline`, `Some(false)` = drop),
/// then the declaration default (keep). Returns `(drop, source)` where
/// `source` is the resolved layer's receipt name (`legacy-env` / `config` /
/// `default` / `legacy-env-rejected`).
fn nested_deadline_dropped(cfg: Option<bool>) -> (bool, &'static str) {
    let (enabled, source) = crate::preset_gates::input_split_nested_deadline_enabled(cfg);
    (!enabled, source)
}

#[cfg(test)]
mod nested_deadline_tests {
    use ny_levers::decls::dark_probes::INPUT_SPLIT_NESTED_DEADLINE;
    use ny_levers::{read_over_config_with, LeverValue, Source};

    /// The pure layering rule this chokepoint depends on, pinned against the
    /// REAL declaration via the `read_over_config_with` test seam (no process
    /// environment mutation). `drop == !value`.
    fn drop_with(env: Option<&str>, cfg: Option<bool>) -> (bool, Source) {
        let resolved = read_over_config_with(
            &INPUT_SPLIT_NESTED_DEADLINE,
            |_| env.map(str::to_owned),
            cfg.map(LeverValue::Bool),
        )
        .expect("Bool config value is always admissible");
        (!resolved.value.as_bool(), resolved.source)
    }

    #[test]
    fn default_keeps_the_nested_deadline() {
        assert_eq!(drop_with(None, None), (false, Source::Default));
    }

    #[test]
    fn preset_false_drops_without_env() {
        // The dist_shift ship shape: `nested_deadline: false` in the preset,
        // zero env.
        assert_eq!(drop_with(None, Some(false)), (true, Source::Config));
    }

    #[test]
    fn preset_true_keeps_explicitly() {
        assert_eq!(drop_with(None, Some(true)), (false, Source::Config));
    }

    #[test]
    fn env_zero_overrides_preset_keep() {
        // The historical diagnostic A/B still works under a keep-preset.
        assert_eq!(drop_with(Some("0"), Some(true)), (true, Source::LegacyEnv));
    }

    #[test]
    fn env_one_overrides_preset_drop() {
        // The kill switch for a shipped relaxation: `=1` restores the
        // deadline-carrying arm even where the preset says drop.
        assert_eq!(
            drop_with(Some("1"), Some(false)),
            (false, Source::LegacyEnv)
        );
    }

    #[test]
    fn malformed_env_suppresses_preset_and_keeps() {
        // A typo'd token must resolve to the SAFE arm (keep the deadline),
        // not silently fall through to the preset relaxation.
        assert_eq!(
            drop_with(Some("true"), Some(false)),
            (false, Source::LegacyEnvRejected)
        );
    }
}
