// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::batched_domain::BatchedDomains;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::graph::DomainSpecCrownResult;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::faer_parallelism::RayonTaskGuard;
use crate::network::collect_intermediate_bounds;
use crate::GraphNetwork;

use super::context::DenseSpecStageTiming;
use super::{
    BatchedBackwardContext, BatchedBackwardMode, BatchedSpecBackwardResult, GpuBetaOptSpec,
};

/// #lsnc-shared-fwd gate. When the input-split batched forward has a SHARED
/// warmup reference-bounds map and no per-domain intermediate refinement is
/// active, every empty-history sub-box reuses that ONE map (borrowed) instead of
/// deep-cloning it per domain. The per-domain node-bounds clone is redundant in
/// that regime because the cloned map is read-only downstream — the tightening
/// comes purely from concretizing the shared linear bounds over each sub-box's
/// smaller INPUT box, never from mutating the intermediate node bounds. So this
/// is a pure allocator-overhead removal, BIT-IDENTICAL to the per-domain clone
/// path. Default ON; set `NY_INPUT_SPLIT_SHARED_FWD=0` to force the historical
/// per-domain clone (the A/B + parity reference).
fn input_split_shared_fwd_enabled() -> bool {
    !matches!(
        std::env::var("NY_INPUT_SPLIT_SHARED_FWD").ok().as_deref(),
        Some("0") | Some("false")
    )
}

/// #lsnc-skip-node-bounds (S3b) gate. The dense-spec batched concretize
/// (`concretize_batched_results_specs`) historically deep-cloned every
/// domain's node-bounds map into `DomainSpecCrownResult.node_bounds`, but the
/// INPUT-SPLIT lane — the only caller of
/// `propagate_crown_batched_with_context_specs_timed`
/// (`input_split/shared_specs.rs`) — reads only `output_bounds` and
/// `input_linear` and drops the map unread. The multi-objective lane
/// (`propagate_crown_with_batched_domains_full_specs_beta_opt`, whose caller
/// `batched_dense_specs.rs` DOES consume `node_bounds`) is NOT gated: it
/// always passes `skip_node_bounds = false`. Skipping an unread clone cannot
/// change any bound, mask, or counter, so the fast path is BIT-IDENTICAL by
/// construction; parity is pinned by
/// `test_input_split_skip_node_bounds_bit_identical_lsnc_s3b`.
/// Default ON (flipped after the parity test + the end-to-end lsnc
/// verdict-identity A/B ran green: instances 0/1/3/5/6 verdicts and
/// per-batch verified/clipped/gap trajectories identical, 2026-07-18);
/// set `NY_INPUT_SPLIT_SKIP_NODE_BOUNDS=0` to force the historical
/// per-domain clone (the A/B + parity reference), mirroring
/// `NY_INPUT_SPLIT_SHARED_FWD`.
static SKIP_NODE_BOUNDS_MODE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// Whether the input-split lane skips the discarded node-bounds clone
/// (see [`SKIP_NODE_BOUNDS_MODE`]).
fn input_split_skip_node_bounds_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match SKIP_NODE_BOUNDS_MODE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = !matches!(
                std::env::var("NY_INPUT_SPLIT_SKIP_NODE_BOUNDS")
                    .ok()
                    .as_deref(),
                Some("0") | Some("false")
            );
            SKIP_NODE_BOUNDS_MODE.store(i8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// #lsnc-batched-interm gate (design-doc slice S2). `-1` = uninitialized
/// (read the env once, then cache); `1` = ON; `0` = OFF.
///
/// Selects the BATCHED intermediate-bounds forward for the input-split
/// no-warmup regime (every domain has `base_bounds = None` and
/// `split_count == 0`, the lsnc `bound_prop_method: crown` configuration):
/// `GraphNetwork::collect_node_bounds_batched` resolves the graph-structure
/// dispatch once per batch and runs the EXACT per-domain kernels of the
/// per-domain reference (`collect_intermediate_bounds`) under coarse-chunked
/// rayon, instead of one fine-grained rayon task + full graph re-resolution
/// per domain. BIT-IDENTICAL parity class: no re-implemented arithmetic, no
/// cross-domain state; proven by
/// `test_batched_interm_bit_identical_to_per_domain_collect` (per-node bounds
/// bits) and `test_input_split_batched_interm_seam_bit_identical` (full
/// dense-spec pipeline bits, gate ON vs OFF). Graphs outside the proven
/// plain-IBP class decline structurally to the untouched reference path
/// (`batched_interm_forward_supported`).
///
/// Default ON: the parity tests above and the end-to-end lsnc verdict-
/// identity check (instances 0,1,3,5,6, gate ON vs OFF — verdicts and
/// result files byte-identical, domain trajectories identical) ran green.
/// Set `NY_INPUT_SPLIT_BATCHED_INTERM=0|false` to force the historical
/// per-domain reference (the A/B + parity leg).
static BATCHED_INTERM_MODE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// Whether the batched intermediate-bounds forward is enabled
/// (see [`BATCHED_INTERM_MODE`]).
fn input_split_batched_interm_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match BATCHED_INTERM_MODE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = !matches!(
                std::env::var("NY_INPUT_SPLIT_BATCHED_INTERM")
                    .ok()
                    .as_deref(),
                Some("0") | Some("false")
            );
            BATCHED_INTERM_MODE.store(i8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// Test-only runtime override for the skip-node-bounds gate: `Some(true|false)`
/// forces ON/OFF, `None` restores the env-derived default. Mirrors
/// `force_batched_relu` so parity tests can A/B without mutating process env.
#[cfg(test)]
pub(crate) fn force_skip_node_bounds(mode: Option<bool>) {
    use std::sync::atomic::Ordering;
    let v = match mode {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    SKIP_NODE_BOUNDS_MODE.store(v, Ordering::Relaxed);
}

/// Test-only runtime override for the batched-interm gate: `Some(true|false)`
/// forces ON/OFF, `None` restores the env-derived default. Mirrors
/// `force_batched_relu` so parity tests can A/B the exact same pipeline
/// without mutating process-global env. Tests MUST restore `None` afterward.
#[cfg(test)]
pub(crate) fn force_batched_interm(mode: Option<bool>) {
    use std::sync::atomic::Ordering;
    let v = match mode {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    BATCHED_INTERM_MODE.store(v, Ordering::Relaxed);
}

/// Test-only probe: number of times the batched-interm forward actually
/// ENGAGED (did not decline). Lets the seam parity test assert which leg ran
/// (checklist Part 3 A, "decline/fallback leg exercised").
#[cfg(test)]
pub(crate) static BATCHED_INTERM_ENGAGED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Test-only lock serializing tests that FORCE the process-global cached
/// gates above while driving the shared dense-spec pipeline. Without it, the
/// S3b parity test's `force_skip_node_bounds(Some(true))` window races the S2
/// seam test running concurrently in another harness thread (its reference
/// leg then sees an empty `node_bounds` map), and vice versa. Any test that
/// calls a `force_*` override in this module MUST hold this lock for its full
/// body. Poison-tolerant: a panicked parity test must not mask the others.
#[cfg(test)]
pub(crate) static SPEC_GATE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
impl BetaCrownVerifier {
    // `delta_seeds` (#cone-delta): the domain's delta pre-nodes, forwarded to
    // `compute_constrained_forward_bounds` on the split-history arm. Dark,
    // `NY_CONE_REFRESH`-gated; `None` keeps full-history seeding.
    fn prepare_dense_spec_forward_domain(
        &self,
        graph: &GraphNetwork,
        input: BoundedTensor,
        history: &GraphSplitHistory,
        base_bounds: Option<&HashMap<String, Arc<BoundedTensor>>>,
        delta_seeds: Option<&[String]>,
        engine: &dyn GemmEngine,
    ) -> Result<(HashMap<String, Arc<BoundedTensor>>, BoundedTensor)> {
        if history.split_count == 0 {
            // Keep the batched dense-spec path aligned with SpecCrownRequest:
            // empty-history domains must reuse precomputed node bounds verbatim,
            // and when no bounds are supplied they must collect intermediates
            // with the same CROWN-IBP-vs-IBP heuristic as the scalar path. #4403
            if let Some(bounds) = base_bounds {
                // #cone-delta increment 2: verbatim reuse of the inherited map
                // is `Arc::clone` per entry — no tensor copies. Only the
                // config-gated IBP-refresh arm below materializes owned
                // tensors (it builds intersected values anyway).
                let cloned_bounds: HashMap<String, Arc<BoundedTensor>> = bounds
                    .iter()
                    .map(|(name, bounds)| (name.clone(), Arc::clone(bounds)))
                    .collect();

                // #cgan-batched-stack: fresh per-subdomain IBP intersected with
                // the shared warmup reference bounds. This restores the
                // per-domain re-anchoring the rayon `ibp_enhancement` path
                // performs (previously omitted here per #4210): without it, all
                // empty-history domains clone one shared map and every
                // per-domain backward computes bit-identical relaxations whose
                // bias gap never shrinks with splitting. SOUND: both maps are
                // valid enclosures of this domain's reachable values, so their
                // elementwise intersection is too (`merge_reference_bound_maps`
                // keeps the current entry on shape mismatch or disjointness).
                // Fail-open to the verbatim clone (today's behavior) on any
                // error — looser but sound.
                if self.config.input_split_batched_ibp_refresh {
                    match graph.collect_node_bounds_with_engine_and_deadline(
                        &input,
                        Some(engine),
                        self.config.alpha_config.deadline,
                    ) {
                        Ok(fresh_ibp) => {
                            // Materialize a plain view for the shared merge
                            // helper (this gated arm deep-materialized before
                            // #cone-delta increment 2 too; values unchanged).
                            let parent_plain: HashMap<String, BoundedTensor> = cloned_bounds
                                .iter()
                                .map(|(name, bounds)| (name.clone(), bounds.as_ref().clone()))
                                .collect();
                            match crate::network::merge_reference_bound_maps(
                                Some(&parent_plain),
                                Some(&fresh_ibp),
                            ) {
                                Ok(Some(refreshed)) => {
                                    return Ok((
                                        refreshed
                                            .into_iter()
                                            .map(|(name, bounds)| (name, Arc::new(bounds)))
                                            .collect(),
                                        input,
                                    ))
                                }
                                Ok(None) => {}
                                Err(err) => {
                                    tracing::debug!(
                                        %err,
                                        "batched per-domain IBP refresh merge failed; using shared reference bounds verbatim"
                                    );
                                }
                            }
                        }
                        Err(err) => {
                            tracing::debug!(
                                %err,
                                "batched per-domain IBP refresh collection failed; using shared reference bounds verbatim"
                            );
                        }
                    }
                }
                return Ok((cloned_bounds, input));
            }

            let collected_bounds = collect_intermediate_bounds(
                graph,
                &input,
                self.config.alpha_config.deadline,
                Some(engine),
            )?
            .into_iter()
            .map(|(name, bounds)| (name, Arc::new(bounds)))
            .collect();
            return Ok((collected_bounds, input));
        }

        self.compute_constrained_forward_bounds(graph, &input, history, base_bounds, delta_seeds)
    }

    /// Propagate CROWN bounds for batched domains using a dense spec matrix.
    ///
    /// Like `propagate_crown_with_batched_domains_full` but accepts a multi-row
    /// `spec_matrix` and returns `DomainSpecCrownResult` preserving per-domain
    /// input `LinearBounds` for split scoring (SB heuristic).
    ///
    /// Part of #4116 Packet A.
    pub fn propagate_crown_with_batched_domains_full_specs(
        &self,
        graph: &GraphNetwork,
        domains: &[&GraphBabDomain],
        batched: &BatchedDomains,
        spec_matrix: &Array2<f32>,
        engine: &dyn GemmEngine,
    ) -> Result<Vec<DomainSpecCrownResult>> {
        Ok(self
            .propagate_crown_with_batched_domains_full_specs_beta_opt(
                graph,
                domains,
                batched,
                spec_matrix,
                engine,
                None,
            )?
            .results)
    }

    /// β-optimizing form of [`propagate_crown_with_batched_domains_full_specs`]
    /// (#w4-split-tightening): threads an optional per-domain β-optimization
    /// request into the GPU resnet fast-path and returns the full
    /// [`BatchedSpecBackwardResult`] (bounds + `optimized_betas` for child β
    /// warm-starting). `beta_opt = None` ⇒ byte-identical single-shot lane.
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_with_batched_domains_full_specs_beta_opt(
        &self,
        graph: &GraphNetwork,
        domains: &[&GraphBabDomain],
        batched: &BatchedDomains,
        spec_matrix: &Array2<f32>,
        engine: &dyn GemmEngine,
        beta_opt: Option<&GpuBetaOptSpec<'_>>,
    ) -> Result<BatchedSpecBackwardResult> {
        if domains.is_empty() {
            return Ok(BatchedSpecBackwardResult {
                results: Vec::new(),
                intermediate_la: None,
                stage_timing: None,
                optimized_betas: None,
                optimized_alphas: None,
                infeasible_domains: None,
            });
        }
        let ctx = BatchedBackwardContext::from_domains(domains, batched)?;
        // Multi-objective lane: `batched_dense_specs.rs` consumes
        // `node_bounds`, so the #lsnc-skip-node-bounds gate never applies here.
        self.batched_forward_then_backward_specs(
            graph,
            &ctx,
            spec_matrix,
            engine,
            BatchedBackwardMode::Standard,
            beta_opt,
            false,
        )
    }

    /// Propagate CROWN bounds for batched domains with lA capture using a dense
    /// spec matrix.
    ///
    /// Like `propagate_crown_with_batched_domains_full_specs` but also captures
    /// intermediate `LinearBounds` at each node for warm-starting subsequent
    /// backward passes on child domains.
    ///
    /// Part of #4116 Packet A.
    pub fn propagate_crown_with_batched_domains_full_specs_capture_la(
        &self,
        graph: &GraphNetwork,
        domains: &[&GraphBabDomain],
        batched: &BatchedDomains,
        spec_matrix: &Array2<f32>,
        engine: &dyn GemmEngine,
    ) -> Result<BatchedSpecBackwardResult> {
        if domains.is_empty() {
            return Ok(BatchedSpecBackwardResult {
                results: Vec::new(),
                intermediate_la: None,
                stage_timing: None,
                optimized_betas: None,
                optimized_alphas: None,
                infeasible_domains: None,
            });
        }
        let ctx = BatchedBackwardContext::from_domains(domains, batched)?;
        self.propagate_crown_batched_with_context_specs_capture_la(graph, &ctx, spec_matrix, engine)
    }

    /// Dense-spec batched CROWN backward propagation.
    ///
    /// Like `propagate_crown_batched_with_context` but accepts a multi-row
    /// `spec_matrix` instead of a scalar `objective` vector. Returns
    /// `DomainSpecCrownResult` which preserves per-domain input `LinearBounds`.
    ///
    /// Part of #4116 Packet A Step 3.
    // Production callers use the `_timed` / `_capture_la` variants; this thin
    // adapter is exercised only by the parity/soundness test suites.
    #[cfg(test)]
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_batched_with_context_specs(
        &self,
        graph: &GraphNetwork,
        ctx: &BatchedBackwardContext,
        spec_matrix: &Array2<f32>,
        engine: &dyn GemmEngine,
    ) -> Result<Vec<DomainSpecCrownResult>> {
        let result =
            self.propagate_crown_batched_with_context_specs_timed(graph, ctx, spec_matrix, engine)?;
        Ok(result.results)
    }

    /// Dense-spec batched CROWN backward propagation with stage timing retained.
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_batched_with_context_specs_timed(
        &self,
        graph: &GraphNetwork,
        ctx: &BatchedBackwardContext,
        spec_matrix: &Array2<f32>,
        engine: &dyn GemmEngine,
    ) -> Result<BatchedSpecBackwardResult> {
        // #lsnc-skip-node-bounds (S3b): this entry's only production caller is
        // the input-split rebound (`input_split/shared_specs.rs`), which reads
        // only `output_bounds` + `input_linear` and drops `node_bounds` unread
        // — so the gate may skip populating it.
        self.batched_forward_then_backward_specs(
            graph,
            ctx,
            spec_matrix,
            engine,
            BatchedBackwardMode::Standard,
            None,
            input_split_skip_node_bounds_enabled(),
        )
    }

    /// Dense-spec batched CROWN backward propagation with lA capture.
    ///
    /// Like `propagate_crown_batched_with_context_specs` but also captures
    /// intermediate `LinearBounds` at each node.
    ///
    /// Part of #4116 Packet A Step 3.
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_batched_with_context_specs_capture_la(
        &self,
        graph: &GraphNetwork,
        ctx: &BatchedBackwardContext,
        spec_matrix: &Array2<f32>,
        engine: &dyn GemmEngine,
    ) -> Result<BatchedSpecBackwardResult> {
        let capture_intermediate = self.config.enable_la_warm_start;
        self.batched_forward_then_backward_specs(
            graph,
            ctx,
            spec_matrix,
            engine,
            BatchedBackwardMode::WithLaCapture {
                histories: &ctx.histories,
                cached_la: &ctx.cached_la,
                capture_intermediate,
            },
            None,
            false,
        )
    }

    /// Core dense-spec batched CROWN backward: forward pass + backward pass.
    ///
    /// Same forward pass as `batched_forward_then_backward`, but delegates to
    /// `propagate_crown_batched_backward_core_specs` for the backward pass.
    ///
    /// `skip_node_bounds` (#lsnc-skip-node-bounds S3b): when true, the
    /// per-domain `DomainSpecCrownResult.node_bounds` map is left EMPTY instead
    /// of deep-cloning the forward cache — legal only for callers that drop
    /// the field unread (the input-split lane).
    #[allow(clippy::too_many_arguments)]
    fn batched_forward_then_backward_specs(
        &self,
        graph: &GraphNetwork,
        ctx: &BatchedBackwardContext,
        spec_matrix: &Array2<f32>,
        engine: &dyn GemmEngine,
        mode: BatchedBackwardMode<'_>,
        beta_opt: Option<&GpuBetaOptSpec<'_>>,
        skip_node_bounds: bool,
    ) -> Result<BatchedSpecBackwardResult> {
        if ctx.is_empty() {
            return Ok(BatchedSpecBackwardResult {
                results: Vec::new(),
                intermediate_la: if matches!(mode, BatchedBackwardMode::WithLaCapture { .. }) {
                    Some(Vec::new())
                } else {
                    None
                },
                stage_timing: None,
                optimized_betas: None,
                optimized_alphas: None,
                infeasible_domains: None,
            });
        }

        let n_domains = ctx.len();
        let plan = graph.dispatch_plan()?;

        tracing::debug!(
            n_domains = n_domains,
            n_layers = plan.exec_order.len(),
            num_specs = spec_matrix.nrows(),
            "Starting dense-spec batched CROWN backward pass"
        );

        // Forward pass — identical to scalar path.
        //
        // #lsnc-shared-fwd: detect the input-split regime where every empty-history
        // sub-box shares ONE warmup reference-bounds map and no per-domain
        // intermediate refinement is active. In that regime the historical
        // per-domain `prepare_dense_spec_forward_domain` clones the same shared map
        // once PER DOMAIN — the dominant allocator pressure at large batches — even
        // though the clone is never mutated downstream (the backward reads it
        // read-only; the tightening comes purely from the per-sub-box input box).
        // So we deref the shared warmup map ONCE and alias it across the batch,
        // carrying only the genuinely per-domain input boxes. BIT-IDENTICAL to the
        // per-domain clone path (same math, one fewer allocation per domain).
        let shared_base: Option<&HashMap<String, Arc<BoundedTensor>>> = {
            let base0 = ctx.base_bounds.first().copied().flatten();
            base0.filter(|b0| {
                input_split_shared_fwd_enabled()
                    && !self.config.input_split_batched_ibp_refresh
                    && ctx.histories.iter().all(|h| h.split_count == 0)
                    && ctx
                        .base_bounds
                        .iter()
                        .all(|entry| matches!(entry, Some(p) if std::ptr::eq(*p, *b0)))
            })
        };

        let forward_start = Instant::now();

        // Owned storage that outlives the borrowed `cache_refs` below.
        // #cone-delta increment 2: the shared-warmup case borrows the warmup
        // map DIRECTLY (its values are already `Arc`-shared) — the historical
        // "deref the arcs once" materialization is gone entirely.
        let shared_map: Option<&HashMap<String, Arc<BoundedTensor>>>;
        let mut per_domain_caches: Vec<HashMap<String, Arc<BoundedTensor>>> = Vec::new();
        let constrained_inputs: Vec<BoundedTensor>;
        let materialize_elapsed_s;

        if let Some(base) = shared_base {
            shared_map = Some(base);
            // split_count==0 ⇒ the constrained box is the domain input verbatim,
            // exactly what `prepare_dense_spec_forward_domain` returns as its second
            // tuple element in this branch.
            let mut inputs = Vec::with_capacity(n_domains);
            for idx in 0..n_domains {
                inputs.push(ctx.batched.input_bounds_at(idx)?);
            }
            constrained_inputs = inputs;
            materialize_elapsed_s = 0.0;
        } else {
            shared_map = None;
            // #lsnc-batched-interm (slice S2): in the no-warmup input-split
            // regime (every domain: base_bounds = None, split_count == 0 — the
            // lsnc plain-CROWN configuration), the reference arm below runs
            // `collect_intermediate_bounds` once per domain under a
            // fine-grained rayon fan-out. The batched collector computes the
            // BIT-IDENTICAL per-node bounds for the whole batch in one pass
            // (graph dispatch resolved once, same per-domain kernels, coarse
            // rayon chunks). Any decline (`None`) falls through to the
            // untouched reference arm.
            #[allow(clippy::type_complexity)]
            let batched_interm: Option<
                Vec<Result<(HashMap<String, Arc<BoundedTensor>>, BoundedTensor)>>,
            > = if input_split_batched_interm_enabled()
                && ctx.base_bounds.iter().all(Option::is_none)
                && ctx.histories.iter().all(|h| h.split_count == 0)
            {
                let per_domain_inputs: Result<Vec<BoundedTensor>> = (0..n_domains)
                    .map(|idx| ctx.batched.input_bounds_at(idx))
                    .collect();
                match per_domain_inputs {
                    // Any malformed input box declines the whole batch to
                    // the reference arm, which reproduces the identical
                    // per-domain error (`input_bounds_at` is deterministic).
                    Err(_) => None,
                    Ok(inputs) => graph
                        .collect_node_bounds_batched(
                            &inputs,
                            Some(engine),
                            self.config.alpha_config.deadline,
                        )
                        .map(|maps| {
                            #[cfg(test)]
                            BATCHED_INTERM_ENGAGED
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            tracing::debug!(
                                n_domains,
                                "batched-interm forward engaged (#lsnc-batched-interm)"
                            );
                            debug_assert_eq!(maps.len(), inputs.len());
                            maps.into_iter()
                                .zip(inputs)
                                .map(|(map, input)| {
                                    map.map(|m| {
                                        // Fresh forward maps: wrap each tensor
                                        // in an Arc by move (no copies).
                                        let m = m
                                            .into_iter()
                                            .map(|(name, bounds)| (name, Arc::new(bounds)))
                                            .collect();
                                        (m, input)
                                    })
                                })
                                .collect()
                        }),
                }
            } else {
                None
            };

            let forward_results: Vec<_> = match batched_interm {
                Some(results) => results,
                None => (0..n_domains)
                    .into_par_iter()
                    .map(|idx| {
                        let _rayon_task_guard = RayonTaskGuard::new();
                        let input = ctx.batched.input_bounds_at(idx)?;
                        self.prepare_dense_spec_forward_domain(
                            graph,
                            input,
                            ctx.histories[idx],
                            ctx.base_bounds[idx],
                            ctx.delta_seeds[idx], // #cone-delta
                            engine,
                        )
                    })
                    .collect(),
            };

            per_domain_caches.reserve(n_domains);
            let mut inputs = Vec::with_capacity(n_domains);
            let materialize_start = Instant::now();
            for (i, result) in forward_results.into_iter().enumerate() {
                match result {
                    Ok((cache, input)) => {
                        per_domain_caches.push(cache);
                        inputs.push(input);
                    }
                    Err(e) if e.is_infeasible_domain() => {
                        return Err(e);
                    }
                    Err(e) => {
                        return Err(NyError::InvalidSpec(format!(
                            "Forward pass failed for domain {}: {}",
                            i, e
                        )));
                    }
                }
            }
            constrained_inputs = inputs;
            materialize_elapsed_s = materialize_start.elapsed().as_secs_f64();
        }
        let forward_elapsed_s = forward_start.elapsed().as_secs_f64();

        // Borrow the caches: shared ⇒ N aliases of ONE map (no per-domain clone);
        // per-domain ⇒ one borrow of each owned map.
        let cache_refs: Vec<&HashMap<String, Arc<BoundedTensor>>> = match shared_map {
            Some(map) => vec![map; n_domains],
            None => per_domain_caches.iter().collect(),
        };

        let backward_start = Instant::now();
        let mut result = self.propagate_crown_batched_backward_core_specs(
            graph,
            n_domains,
            plan,
            &cache_refs,
            &constrained_inputs,
            &ctx.beta_states,
            &ctx.alpha_states,
            spec_matrix,
            engine,
            mode,
            ctx.mul_binary_alphas, // #4284: thread shared MulBinary alphas
            beta_opt,              // #w4-split-tightening: per-domain β ascent request
            skip_node_bounds,      // #lsnc-skip-node-bounds S3b
        )?;
        let backward_elapsed_s = backward_start.elapsed().as_secs_f64();

        result.stage_timing = Some(DenseSpecStageTiming {
            forward_elapsed_s,
            backward_elapsed_s,
            materialize_elapsed_s,
        });
        Ok(result)
    }
}
