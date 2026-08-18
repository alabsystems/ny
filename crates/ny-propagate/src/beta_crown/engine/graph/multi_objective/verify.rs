// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Multi-objective graph β-CROWN verification.
//!
//! Verifies both disjunctive (OR) and conjunctive (AND) properties by running all
//! objectives through a single branch-and-bound pass. All objectives share the same
//! domain queue and branching decisions, avoiding redundant BaB searches.
//!
//! - **Disjunctive**: ALL constraints must be proved violated → SAFE.
//! - **Conjunctive**: ANY constraint proved violated → conjunction impossible → SAFE.
//!
//! Key types: [`MultiObjectiveTargets`] holds the objective/threshold vectors,
//! [`MultiObjectiveGraphBabDomain`] extends the standard domain with per-objective bounds.
//!
//! Entry points:
//! - `BetaCrownVerifier::verify_graph_relu_split_multi_objective` (disjunctive)
//! - `BetaCrownVerifier::verify_graph_relu_split_multi_objective_conjunctive_with_engine` (conjunctive)

use std::collections::BinaryHeap;
use std::ffi::OsStr;
use std::time::Instant;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::instrument;

use crate::beta_crown::result::BetaCrownResult;
use crate::layers::Layer;
use crate::{GraphNetwork, OwnedSignNormalizedObjectiveSet};

use super::super::super::BetaCrownVerifier;
use super::super::domain_batch::{
    GraphDomainBatchEmitTiming, GraphDomainBatchExecutionMode, GraphDomainBatchExecutor,
    GraphDomainBatchPlan, MultiObjectiveBatchRequest,
};
use super::super::shared::state::GraphBabLifecycle;
use super::bounded_shared_executor::{
    admit as admit_bounded_shared_executor, bounded_frontier_domain_limit,
    graph_may_support_bounded_beta, resolve_gate as resolve_bounded_shared_executor_gate,
    workload_may_support_bounded_beta, DeadlineCpuGemmEngine, MO_CUDA_BOUNDED_SHARED_EXECUTOR_ENV,
};
use super::finalize::{finalize_multi_objective_result, resolve_multi_objective_loop_boundary};
use super::finalized_root_handoff::ResidentBabFinalizedRootHandoffV1;
use super::queue::{
    apply_batched_results, pop_domain_batch, prefilter_batch, LeafOracleCtx,
    MultiObjectiveBatchApplyStatus,
};
use super::root::{
    evaluate_root, validate_multi_objective_inputs, MultiObjectiveProperty,
    MultiObjectiveRootEvaluation, MultiObjectiveRootOutcome, MultiObjectiveRootRequest,
    MultiObjectiveRootState,
};
use super::sequential::{SequentialMultiObjectiveBatchStatus, SequentialMultiObjectiveContext};
use super::stall_obbt_canary::StallObbtCanary;

/// Finish one shared-executor batch without allowing optional metrics work to
/// delay or replace an already-typed deadline outcome. Metrics remain fallible
/// for ordinary completed batches.
fn finish_shared_multi_objective_batch<F>(
    status: MultiObjectiveBatchApplyStatus,
    lifecycle: &mut GraphBabLifecycle,
    cuts_generated: usize,
    emit_metrics: F,
) -> Result<Option<BetaCrownResult>>
where
    F: FnOnce() -> Result<()>,
{
    if status == MultiObjectiveBatchApplyStatus::DeadlineExpired {
        lifecycle.cuts_generated = cuts_generated;
        return Ok(Some(lifecycle.timeout_result()));
    }
    emit_metrics()?;
    Ok(None)
}

/// Convert a failed bounded-executor publication poll into the verifier's
/// graceful timeout result. This barrier is needed after queue mutation and
/// optional metrics because a clean drained frontier intentionally takes
/// precedence over a timeout first noticed on the next loop iteration.
fn poll_bounded_shared_publication(
    active: bool,
    engine: Option<&dyn GemmEngine>,
    lifecycle: &mut GraphBabLifecycle,
    cuts_generated: usize,
) -> Result<Option<BetaCrownResult>> {
    if !active {
        return Ok(None);
    }
    let engine = engine.ok_or_else(|| {
        ny_core::NyError::InvalidSpec(
            "active bounded shared executor has no deadline authority".into(),
        )
    })?;
    match engine.poll_crown_backward_deadline() {
        Ok(()) => Ok(None),
        Err(error) if error.is_deadline_exceeded() => {
            lifecycle.cuts_generated = cuts_generated;
            Ok(Some(lifecycle.timeout_result()))
        }
        Err(error) => Err(error),
    }
}

const MO_CUDA_FACTORY_ENGINE_HANDOFF_ENV: &str = "NY_MO_CUDA_FACTORY_ENGINE_HANDOFF";

/// Resolve the typed post-root handoff policy with its exact environment
/// override. An absent environment value inherits the typed config; literal
/// `1` enables, while every other present byte string disables.
#[inline]
fn resolve_mo_cuda_factory_engine_handoff_gate(
    typed_enabled: bool,
    raw_env: Option<&OsStr>,
) -> bool {
    raw_env.map_or(typed_enabled, |raw| raw == OsStr::new("1"))
}

#[inline]
fn post_root_factory_engine_is_eligible(engine: &dyn GemmEngine) -> bool {
    engine.supports_deadline_safe_post_root_multi_objective_bab()
        && engine.as_gpu_crown_backward().is_some_and(|gpu| {
            gpu.provides_sound_gpu_crown() && gpu.honors_crown_backward_deadline()
        })
}

#[inline]
const fn post_root_cuts_enabled(configured: bool, bounded_shared_active: bool) -> bool {
    configured && !bounded_shared_active
}

/// Resolve the engine used strictly after root evaluation.
///
/// The caller's already-resolved engine always wins and retains the exact
/// legacy route. Only a live, explicitly armed finite authority may observe the
/// preinitialized factory slot. The injected accessor is get-only in
/// production; keeping it behind `FnOnce` also makes a second observation
/// structurally impossible.
fn resolve_post_root_multi_objective_engine<'caller, 'factory>(
    caller_engine: Option<&'caller dyn GemmEngine>,
    handoff_enabled: bool,
    deadline: Instant,
    now: impl FnOnce() -> Instant,
    preinitialized_engine: impl FnOnce() -> Option<&'factory dyn GemmEngine>,
) -> Option<&'caller dyn GemmEngine>
where
    'factory: 'caller,
{
    if caller_engine.is_some() {
        return caller_engine;
    }
    if !handoff_enabled || now() >= deadline {
        return None;
    }

    preinitialized_engine().filter(|engine| post_root_factory_engine_is_eligible(*engine))
}

impl BetaCrownVerifier {
    /// Multi-objective verification for disjunctive properties.
    ///
    /// Verifies ALL objectives simultaneously in a single BaB pass, sharing
    /// computation across objectives. For disjunctive properties (OR), this is
    /// required: the property is SAFE only if ALL constraints are proved violated.
    ///
    /// # Arguments
    /// * `graph` - The DAG-based neural network
    /// * `input` - Input bounds
    /// * `objectives` - List of objective vectors (each is a linear combination of outputs)
    /// * `thresholds` - Threshold for each objective (usually all 0.0)
    ///
    /// # Returns
    /// * `Verified` if ALL objectives are verified (all lower > threshold)
    /// * `Unknown` if ANY objective cannot be verified within timeout
    #[instrument(skip(self, graph, input, objectives), fields(num_objectives = objectives.len(), input_shape = ?input.shape()))]
    pub fn verify_graph_relu_split_multi_objective(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
    ) -> Result<BetaCrownResult> {
        self.verify_graph_relu_split_multi_objective_with_engine(
            graph,
            input,
            objectives,
            thresholds,
            self.engine(),
            None,
        )
    }

    /// Multi-objective Graph β-CROWN verification with GPU acceleration.
    ///
    /// Same as `verify_graph_relu_split_multi_objective` but with optional GPU engine
    /// for accelerated bound computation. Uses disjunctive semantics (ALL must verify).
    ///
    /// `deadline`: If `Some`, the BaB engine derives its phase budgets from
    /// remaining wall-clock time instead of `self.config.timeout` (#4321).
    #[instrument(skip(self, graph, input, objectives, engine, deadline), fields(num_objectives = objectives.len(), input_shape = ?input.shape()))]
    pub fn verify_graph_relu_split_multi_objective_with_engine(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        let engine = self.resolve_engine(engine);
        self.verify_graph_relu_split_multi_objective_core(
            graph,
            input,
            MultiObjectiveProperty::borrowed(objectives, thresholds),
            engine,
            false, // conjunctive=false → disjunctive semantics
            deadline,
        )
    }

    /// Consume one move-owned sign-normalized property for disjunctive graph
    /// ReLU-split verification.
    ///
    /// This opt-in ingress retains the exact non-`Clone` objective/threshold
    /// owner through root evaluation. It does not observe retained-v1
    /// provenance, query a resident provider, compose a static payload, or
    /// alter runtime selection; today it returns the same ordinary verification
    /// result as [`Self::verify_graph_relu_split_multi_objective_with_engine`].
    /// Existing borrowed entry points remain the default. An error is terminal
    /// for this consuming call and drops the still-intact owner.
    ///
    /// The property is consumed and cannot be reused after this call:
    ///
    /// ```compile_fail
    /// use ny_core::GemmEngine;
    /// use ny_propagate::{
    ///     BetaCrownVerifier, BoundedTensor, GraphNetwork,
    ///     OwnedSignNormalizedObjectiveSet,
    /// };
    /// use std::time::Instant;
    ///
    /// fn consume_once(
    ///     verifier: &BetaCrownVerifier,
    ///     graph: &GraphNetwork,
    ///     input: &BoundedTensor,
    ///     property: OwnedSignNormalizedObjectiveSet,
    ///     engine: Option<&dyn GemmEngine>,
    ///     deadline: Option<Instant>,
    /// ) {
    ///     let _ = verifier.verify_graph_relu_split_multi_objective_owned_with_engine(
    ///         graph, input, property, engine, deadline,
    ///     );
    ///     let _ = property.rows();
    /// }
    /// ```
    #[instrument(skip(self, graph, input, property, engine, deadline), fields(num_objectives = property.len(), input_shape = ?input.shape()))]
    pub fn verify_graph_relu_split_multi_objective_owned_with_engine(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        property: OwnedSignNormalizedObjectiveSet,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        let engine = self.resolve_engine(engine);
        self.verify_graph_relu_split_multi_objective_core(
            graph,
            input,
            MultiObjectiveProperty::owned(property),
            engine,
            false,
            deadline,
        )
    }

    /// Multi-objective verification for conjunctive (AND) properties.
    ///
    /// Verifies all AND conjuncts jointly in a single BaB pass. A subdomain is
    /// safe (verified) if ANY single conjunct is proven impossible — the
    /// conjunction cannot hold. This is strictly more powerful than per-constraint
    /// decomposition which requires each conjunct to be universally false.
    ///
    /// # Soundness argument
    ///
    /// CROWN computes lower bounds: `lb_i ≤ spec_i(y)` for all `y` in subdomain.
    /// If `lb_i > threshold_i` for some spec `i`, then `spec_i(y) > threshold_i`
    /// for all `y` → constraint `Cᵢ` cannot hold → conjunction cannot hold.
    /// Marking the subdomain verified when at least one such `i` exists is sound.
    /// "Verified" overall requires every subdomain in the partition to be verified,
    /// covering the entire input space.
    ///
    /// Reference: alpha-beta-CROWN `stop_criterion_batch_any` in
    /// `auto_LiRPA/utils.py:107-113`, `multi_spec_keep_func_all` in
    /// `auto_LiRPA/utils.py:143-144`.
    /// `deadline`: If `Some`, the BaB engine derives its phase budgets from
    /// remaining wall-clock time instead of `self.config.timeout` (#4321).
    #[instrument(skip(self, graph, input, objectives, engine, deadline), fields(num_objectives = objectives.len(), input_shape = ?input.shape()))]
    pub fn verify_graph_relu_split_multi_objective_conjunctive_with_engine(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        let engine = self.resolve_engine(engine);
        self.verify_graph_relu_split_multi_objective_core(
            graph,
            input,
            MultiObjectiveProperty::borrowed(objectives, thresholds),
            engine,
            true, // conjunctive=true → AND semantics
            deadline,
        )
    }

    /// Core multi-objective BaB verification, parameterized by `conjunctive`.
    ///
    /// - `conjunctive=false` (disjunctive): domain verified when ALL objectives
    ///   verified, dropped when ANY violated.
    /// - `conjunctive=true` (conjunctive): domain verified when ANY objective
    ///   verified, dropped when ALL violated.
    fn verify_graph_relu_split_multi_objective_core(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        property: MultiObjectiveProperty<'_>,
        engine: Option<&dyn GemmEngine>,
        conjunctive: bool,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        // Validate before any objective/root shortcut can acquire verdict
        // authority; the shared bootstrap remains a defense-in-depth check.
        self.config.validate()?;
        // This shared graph engine is defined over sign-normalized lower-bound
        // objectives: every root close, domain aggregation, and child close
        // proves `lower > threshold`.  Accepting the legacy scalar engine's
        // `verify_upper_bound` switch here would reverse that meaning without
        // reversing the multi-objective stopping rules.  In particular, a
        // positive constant output could be falsely reported Verified for an
        // upper-bound property whose threshold is below that constant.  CLI
        // callers normalize upper constraints into lower-bound rows before
        // reaching this API; direct Rust callers must do the same.
        if self.config.verify_upper_bound {
            return Err(ny_core::NyError::InvalidSpec(
                "graph multi-objective verification requires sign-normalized lower-bound \
                 objectives (verify_upper_bound=false)"
                    .to_string(),
            ));
        }
        let configured_graph = self.configured_graph_for_crown(graph);
        // Preserve the historical budget boundary: graph configuration is
        // setup work and is not charged to the BaB lifecycle.
        let now = Instant::now();
        let mut lifecycle = GraphBabLifecycle::new(now);
        {
            let (objectives, thresholds) = property.views();
            validate_multi_objective_inputs(objectives, thresholds)?;
        }
        // A caller-supplied boundary is already the CLI ledger's reserved BaB
        // slice. Keep that exact boundary authoritative for optional clipping
        // during both root evaluation and the outer BaB loop.
        let pgd_frac = self
            .config
            .phase_budget
            .post_bab_pgd_fraction
            .clamp(0.0, 0.5);
        let bab_timeout = match deadline {
            Some(dl) => dl.saturating_duration_since(now),
            None => self.config.timeout.mul_f32(1.0 - pgd_frac),
        };
        let bab_deadline = lifecycle.deadline(bab_timeout);
        let attribution_deadline =
            Some(deadline.map_or(bab_deadline, |caller| caller.min(bab_deadline)));
        // Attribution publication is process-global because KFSB prepares
        // domains on Rayon workers. Its owner spans root evaluation and every
        // BaB/KFSB consumer. Invalid calls above cannot consume a prior, and a
        // previous valid owner's Drop has already cleared its publication.
        let _attribution_run =
            crate::network::gap_attribution::attribution_run_guard(attribution_deadline)?;
        let _complete_clip_deadline = self.complete_clip_deadline_overrides.scoped(Some(
            GraphBabLifecycle::fail_closed_deadline(now, bab_timeout),
        ));
        let MultiObjectiveRootEvaluation { outcome, property } = evaluate_root(
            MultiObjectiveRootRequest {
                verifier: self,
                graph: &configured_graph,
                input,
                property,
                engine,
                conjunctive,
                deadline,
            },
            &mut lifecycle,
        )?;
        let root = match outcome {
            MultiObjectiveRootOutcome::Finished(result) => return Ok(*result),
            MultiObjectiveRootOutcome::Continue(root) => *root,
        };
        let (configured_graph, root, property) = match property {
            MultiObjectiveProperty::Owned(property) => {
                let (configured_graph, root, property) =
                    ResidentBabFinalizedRootHandoffV1::new(configured_graph, input, root, property)
                        .into_legacy_parts();
                (
                    configured_graph,
                    root,
                    MultiObjectiveProperty::owned(property),
                )
            }
            borrowed @ MultiObjectiveProperty::Borrowed { .. } => {
                (configured_graph, root, borrowed)
            }
        };
        let graph = &configured_graph;
        let MultiObjectiveRootState {
            initial_output,
            mut root_domain,
            mut selective_root_alpha_candidate,
            relu_nodes,
            mut cut_pool,
            use_batched_gpu: root_use_batched_gpu,
        } = root;
        // The execution-dark handoff proved custody of the exact root output.
        // The unchanged CPU loop never consumed that enclosure, so restore its
        // historical lifetime before any post-root engine/admission decision.
        drop(initial_output);
        let (objectives, thresholds) = property.views();
        // Root evaluation deliberately used only the original resolved caller
        // engine. After it completes, resolve the optional handoff once against
        // the same authoritative BaB deadline used by the outer loop. A cold
        // factory, expired authority, dark gate, or refused capability returns
        // the original `None` and therefore the exact historical sequential
        // route.
        let handoff_env = std::env::var_os(MO_CUDA_FACTORY_ENGINE_HANDOFF_ENV);
        let handoff_enabled = resolve_mo_cuda_factory_engine_handoff_gate(
            self.config.mo_cuda_factory_engine_handoff,
            handoff_env.as_deref(),
        );
        let caller_engine_present = engine.is_some();
        let post_root_engine = resolve_post_root_multi_objective_engine(
            engine,
            handoff_enabled,
            bab_deadline,
            Instant::now,
            crate::sound_f64_gemm::preinitialized_sound_gpu_engine,
        );
        // A second, narrower route may activate the shared executor without
        // lending generic work the CUDA GemmEngine. Admission observes only the
        // already-preinitialized wide GPU capability and requires its audited
        // call-local `2..=8` row β-CROWN surface. The executor itself receives
        // the local deadline-polling CPU facade below; constrained per-child
        // propagation independently reaches CUDA through the existing bounded-
        // β selector. Dark/cold/expired/ineligible calls retain exact `None`.
        let bounded_shared_env = std::env::var_os(MO_CUDA_BOUNDED_SHARED_EXECUTOR_ENV);
        let bounded_shared_enabled = resolve_bounded_shared_executor_gate(
            self.config.mo_cuda_bounded_shared_executor,
            bounded_shared_env.as_deref(),
        );
        let configured_batch_size = self.config.batch_size.max(1);
        let mut bounded_frontier_limit = None;
        let bounded_shared_admission = admit_bounded_shared_executor(
            bounded_shared_enabled,
            caller_engine_present,
            post_root_engine.is_some(),
            configured_batch_size > 1 && !conjunctive,
            bab_deadline,
            Instant::now,
            || {
                if !cut_pool.is_empty()
                    || !graph_may_support_bounded_beta(graph, input, &root_domain.node_bounds)
                    || !workload_may_support_bounded_beta(
                        graph,
                        input,
                        &root_domain.node_bounds,
                        objectives,
                        thresholds,
                        root_domain.verified(),
                    )
                {
                    return false;
                }
                bounded_frontier_limit = bounded_frontier_domain_limit(
                    graph,
                    input,
                    &root_domain.node_bounds,
                    objectives.len(),
                );
                bounded_frontier_limit.is_some()
            },
            crate::sound_gpu_gate::preinitialized_sound_gpu_crown_for_wide,
        );
        bounded_shared_admission.report();
        let bounded_shared_capacity = bounded_shared_admission.accepted_capacity();
        let bounded_shared_active = bounded_shared_capacity.is_some();
        let cuts_enabled = post_root_cuts_enabled(self.config.enable_cuts, bounded_shared_active);
        let effective_max_domains = bounded_shared_capacity.map_or(self.config.max_domains, |_| {
            self.config.max_domains.min(
                bounded_frontier_limit
                    .expect("accepted bounded executor has a frontier-domain limit"),
            )
        });
        let bounded_shared_proxy =
            bounded_shared_capacity.map(|_| DeadlineCpuGemmEngine::new(bab_deadline));
        // The bounded lane's pre-GEMM host work (unstable discovery,
        // BatchedDomains stacking, and union-spec construction) is part of the
        // same authority. Never pop more domains than the backend's audited
        // K<=8 transactional capacity when this facade selected the executor.
        let batch_size = bounded_shared_capacity.map_or(configured_batch_size, |capacity| {
            configured_batch_size.min(capacity)
        });
        let engine: Option<&dyn GemmEngine> = post_root_engine.or_else(|| {
            bounded_shared_proxy
                .as_ref()
                .map(|proxy| proxy as &dyn GemmEngine)
        });
        if bounded_shared_capacity.is_some() {
            // Use the compact heuristic-alpha frontier before queue insertion;
            // dense lA/per-disjunct/shared-alpha warm starts would otherwise be
            // deep-cloned into both children outside the facade's allocation
            // and polling surface.
            root_domain.prepare_for_bounded_executor();
            selective_root_alpha_candidate = None;
        }

        let use_batched_gpu = engine.is_some() && batch_size > 1 && !conjunctive;
        debug_assert!(
            !root_use_batched_gpu || use_batched_gpu,
            "post-root engine resolution must not discard the root's caller engine"
        );
        // Default-dark, measurement-only NeuralSAT-style stall observer. `None`
        // means the loop does not inspect the frontier for this canary. The
        // observer owns only fixed-size counters and cannot mutate the queue,
        // domains, bounds, scheduling, or verdict state.
        let mut stall_obbt_canary = if bounded_shared_active {
            None
        } else {
            StallObbtCanary::from_env()
        };

        // Conflict-clause learning, graph port (win-plan arc C, v2): per-run
        // store, gated NY_BAB_CLAUSE_LEARN=1 (default OFF => disabled store =>
        // byte-identical loop). Scope of THIS store: one graph, one root input
        // box, one (objectives, thresholds, conjunctive) tuple — a clause is
        // only ever recorded from a domain closed verified under the SAME
        // objective semantics it prunes for (see `prefilter_batch` /
        // `apply_batched_results` for the per-close argument, and
        // `conflict_clauses_graph` for the region-inclusion + purity-guard
        // argument). Both batched and sequential exact-leaf closes record at
        // their sole terminal disposition point; the established bound-close
        // sites remain unchanged.
        // This production caller's shared multi-objective close semantics are
        // lower-bound-only. `verify_upper_bound` reaches legacy code paths that
        // do not provide the uniform source authority required by graph clause
        // recording, so fail closed before constructing either ordinary or
        // replay clause state.
        let mut clause_store = if bounded_shared_active || self.config.verify_upper_bound {
            crate::beta_crown::conflict_clauses_graph::GraphClauseStore::disabled()
        } else {
            crate::beta_crown::conflict_clauses_graph::GraphClauseStore::from_env()
        };
        // Proof-replayed clause shortening (default OFF). The established
        // one-deletion policy and its independently default-dark BICCOS-Q
        // Stage-1 proposal share one trusted replay/token/store boundary.
        // Build the immutable run identity and retain the root enclosures only
        // when the ordinary graph-clause path is already armed. Gate-off runs
        // keep the ordinary unbound store and never clone root map/objectives.
        let mut clause_replay = if clause_store.is_enabled() {
            match crate::beta_crown::conflict_clause_replay::GraphClauseReplayRuntime::from_env(
                graph,
                &root_domain.input_bounds,
                &root_domain.node_bounds,
                objectives,
                thresholds,
                conjunctive,
                lifecycle.deadline(bab_timeout),
            ) {
                Ok(Some(runtime)) => {
                    let cap = clause_store
                        .capacity_for_replay_binding()
                        .expect("enabled graph clause store has a positive capacity");
                    clause_store = crate::beta_crown::conflict_clauses_graph::GraphClauseStore::
                        with_capacity_and_replay_run(
                            true,
                            cap,
                            runtime.run_fingerprint().clone(),
                        );
                    tracing::info!(
                        "Graph clause replay armed (NY_BAB_CLAUSE_REPLAY=1; \
                         attempts=16 total=2s per_attempt=250ms)"
                    );
                    Some(runtime)
                }
                Ok(None) => None,
                Err(refusal) => {
                    tracing::info!(
                        ?refusal,
                        "Graph clause replay initialization refused; ordinary clauses unchanged"
                    );
                    None
                }
            }
        } else {
            None
        };
        // BICCOS-Q Stage-0 is a separate, default-off shadow observer. Unlike
        // replay authority, it does not require ordinary clause learning: it
        // receives only immutable child histories/β state and cannot publish
        // a clause, cut, queue mutation, bound, or verdict. Upper-bound mode is
        // outside the lower-bound replay source contract and stays dark.
        let mut biccos_q_stage0 = if bounded_shared_active || self.config.verify_upper_bound {
            None
        } else {
            crate::beta_crown::biccos_q_stage0::BiccosQStage0Telemetry::from_env()
        };
        if clause_store.is_enabled() {
            tracing::info!(
                "Graph multi-objective BaB conflict-clause learning enabled (NY_BAB_CLAUSE_LEARN=1)"
            );
        }

        // DIAGNOSTIC (NY_ACASXU_PROF): the multi-objective BaB explored/verified/queue
        // trajectory. The single-objective β-CROWN loop (engine/core.rs) already honors
        // this env, but the multi-objective graph loop (the cifar100/tinyimagenet resnet
        // path) had no equivalent. Print-only, env-gated — never mutates a bound or a
        // verdict. Also emits the root failing-objective set (which of the N disjuncts is
        // still unverified at the root, worst-first) so an A/B can see which constraint BaB
        // must close and whether the queue is shrinking or exploding.
        let prof = std::env::var("NY_ACASXU_PROF").is_ok();
        if prof {
            let n = root_domain.objective_bounds.len();
            let mut failing: Vec<(usize, f32)> = root_domain
                .objective_bounds
                .iter()
                .enumerate()
                .filter(|(i, _)| !root_domain.verified.get(*i).copied().unwrap_or(false))
                .map(|(i, (lo, _))| (i, *lo))
                .collect();
            failing.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let verified_root = n - failing.len();
            let show: Vec<(usize, f32)> = failing.iter().take(12).copied().collect();
            eprintln!(
                "[NY_ACASXU_PROF] MULTI-OBJ root: verified={verified_root}/{n} conjunctive={conjunctive} n_failing={} worst_failing(obj,lower)={:?}",
                failing.len(),
                show
            );
        }
        let mut last_tick = Instant::now();

        // NY_MO_BAB_TRACE (dark, diagnostic-only): the global-bound-over-wall-clock
        // progress trace for THIS multi-objective batched BaB loop — the lane that
        // `select_graph_branch_kfsb_multi_batched` / `select_graph_branch_multi`
        // feed (NY_MO_KFSB). Per-wave kFSB picks are already visible via
        // NY_MO_KFSB_PROBE, but there was no global bound-vs-time curve, so an
        // "equal verified-count" A/B could not separate "selection doesn't help"
        // from "helps but the α-CROWN preamble masks it". This emits, on a cheap
        // cadence, a one-line `[bab-trace]` record with the worst still-unverified
        // objective LB across open domains. Independent of NY_MO_KFSB: the A/B runs
        // it with and without kFSB and compares the two bound-vs-time curves.
        //
        // Byte-identical when unset: `bab_trace` short-circuits the whole block
        // (including the O(domains × objectives) min-reduction and the wall-clock
        // read), so the disarmed path adds only a per-wave bool test — no stderr
        // write, no allocation, no reduction. Cadence: emit when the last trace was
        // >= 500ms ago OR after `NY_MO_BAB_TRACE_WAVES` waves (default 20), plus one
        // anchor line on the first wave.
        let bab_trace = std::env::var("NY_MO_BAB_TRACE").ok().as_deref() == Some("1");
        let bab_trace_waves = if bab_trace {
            std::env::var("NY_MO_BAB_TRACE_WAVES")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&k| k > 0)
                .unwrap_or(20)
        } else {
            0
        };
        let mut last_bab_trace: Option<Instant> = None;
        let mut waves_since_bab_trace = 0usize;

        let mut queue = BinaryHeap::new();
        queue.push(root_domain);
        let mut batch_index = 0usize;

        // NY_LOOSENESS_PROBE (dark, diagnostic-only): localize the ~0.25 relaxation
        // looseness. When set, at each pop the frontier-worst subdomain is `batch[0]`
        // (max-priority = min-margin). Once it reaches the target depth we sample many
        // inputs in its box (respecting ReLU splits by rejection), forward the TRUE
        // network, and dump per-node NY[l,u] vs true[min,max] looseness. Print-only,
        // never mutates a bound/verdict. See run_looseness_probe below.
        let looseness_probe = !bounded_shared_active
            && std::env::var("NY_LOOSENESS_PROBE").ok().as_deref() == Some("1");
        let mut looseness_worst_margin = f32::INFINITY;
        let mut looseness_dumps_done = 0usize;
        let looseness_max_dumps = std::env::var("NY_LOOSENESS_MAX_DUMPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);

        loop {
            lifecycle.cuts_generated = cut_pool.total_generated;
            if prof && last_tick.elapsed().as_secs_f64() >= 2.0 {
                eprintln!(
                    "[NY_ACASXU_PROF] tick: batch={batch_index} explored={} verified={} queue={} max_depth={} elapsed={:.1}s",
                    lifecycle.domains_explored,
                    lifecycle.domains_verified,
                    queue.len(),
                    lifecycle.max_depth_reached,
                    lifecycle.start_time.elapsed().as_secs_f64(),
                );
                last_tick = Instant::now();
            }

            // NY_MO_BAB_TRACE global bound-vs-wall progress trace (see init above).
            // Disarmed => `bab_trace` is false => the entire block (including the
            // min-reduction over open domains) is skipped => byte-identical.
            if bab_trace {
                waves_since_bab_trace += 1;
                let due = match last_bab_trace {
                    None => true, // first-wave anchor
                    Some(t) => {
                        t.elapsed().as_millis() >= 500 || waves_since_bab_trace >= bab_trace_waves
                    }
                };
                if due {
                    let worst = worst_unverified_objective_lb(
                        queue.iter().map(|d| (d.objective_bounds(), d.verified())),
                    );
                    eprintln!(
                        "{}",
                        format_bab_trace_line(
                            lifecycle.start_time.elapsed().as_secs_f64(),
                            worst,
                            queue.len(),
                            lifecycle.max_depth_reached,
                            lifecycle.domains_explored,
                            lifecycle.has_unresolved(),
                        )
                    );
                    last_bab_trace = Some(Instant::now());
                    waves_since_bab_trace = 0;
                }
            }
            // Resolve a cleanly drained proof before applying the next-iteration
            // wall/domain cap. The helper still gives an unresolved empty
            // frontier deadline precedence, so dropped work cannot become a
            // false proof.
            let frontier_empty = queue.is_empty();
            if let Some(result) = resolve_multi_objective_loop_boundary(
                &lifecycle,
                frontier_empty,
                bab_timeout,
                effective_max_domains,
            ) {
                if prof {
                    if frontier_empty && !lifecycle.has_unresolved() {
                        eprintln!(
                            "[NY_ACASXU_PROF] queue_drained: explored={} verified={} queue={} max_depth={} elapsed={:.2}s",
                            lifecycle.domains_explored,
                            lifecycle.domains_verified,
                            queue.len(),
                            lifecycle.max_depth_reached,
                            lifecycle.start_time.elapsed().as_secs_f64(),
                        );
                    } else {
                        eprintln!(
                            "[NY_ACASXU_PROF] terminate(timeout/limit): explored={} verified={} queue={} max_depth={} elapsed={:.2}s",
                            lifecycle.domains_explored,
                            lifecycle.domains_verified,
                            queue.len(),
                            lifecycle.max_depth_reached,
                            lifecycle.start_time.elapsed().as_secs_f64(),
                        );
                    }
                }
                return Ok(result);
            }

            // Sample the exact frontier the existing pop will consume. This is
            // an immutable, O(1) observation after the deadline/domain boundary
            // check; it returns no decision and cannot affect the pop.
            if let Some(canary) = stall_obbt_canary.as_mut() {
                canary.observe_queue(&queue, batch_size);
            }
            let batch = pop_domain_batch(&mut queue, batch_size);
            if batch.is_empty() {
                break;
            }
            // NY_UNSTABLE_COUNT (dark, diagnostic-only): for the frontier-WORST subdomain
            // count unstable ReLU pre-activation neurons (l<0<u) = the # binaries an exact
            // MILP leaf-verifier would face. Cheap (no sampling). Prints once per new-min
            // margin so we see the count at the plateau depth. Print-only, sound.
            if !bounded_shared_active
                && std::env::var("NY_UNSTABLE_COUNT").ok().as_deref() == Some("1")
            {
                if let Some(worst) = batch.first() {
                    let margin = worst
                        .objective_bounds()
                        .iter()
                        .map(|(l, _)| *l)
                        .fold(f32::INFINITY, f32::min);
                    let mut total = 0usize;
                    let mut per_relu: Vec<(String, usize)> = Vec::new();
                    if let Ok(order) = graph.exec_order() {
                        for name in order {
                            let Some(node) = graph.node(name) else {
                                continue;
                            };
                            if !matches!(node.layer(), Layer::ReLU(_)) {
                                continue;
                            }
                            let Some(pre) = node.inputs().first() else {
                                continue;
                            };
                            let Some(bt) = worst.node_bounds().get(pre) else {
                                continue;
                            };
                            let lo = bt.lower();
                            let hi = bt.upper();
                            let unstable = lo
                                .iter()
                                .zip(hi.iter())
                                .filter(|(&l, &u)| l < 0.0 && u > 0.0)
                                .count();
                            if unstable > 0 {
                                per_relu.push((name.clone(), unstable));
                            }
                            total += unstable;
                        }
                    }
                    eprintln!(
                        "[unstable-count] depth={} margin={:.4} TOTAL_unstable_binaries={} per_relu={:?}",
                        worst.depth(),
                        margin,
                        total,
                        per_relu
                    );
                }
            }

            // NY_LOOSENESS_PROBE dark diagnostic (see tracker init above).
            if looseness_probe && looseness_dumps_done < looseness_max_dumps {
                if let Some(worst) = batch.first() {
                    let margin = worst
                        .objective_bounds()
                        .iter()
                        .map(|(l, _)| *l)
                        .fold(f32::INFINITY, f32::min);
                    let want_depth = std::env::var("NY_LOOSENESS_DEPTH")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok());
                    let min_depth = std::env::var("NY_LOOSENESS_MIN_DEPTH")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(6);
                    let depth_ok = match want_depth {
                        Some(d) => worst.depth() == d,
                        None => worst.depth() >= min_depth,
                    };
                    // Dump when this is a new frontier-worst (min margin) at the target
                    // depth — the last dump is the actual plateau worst subdomain.
                    if depth_ok && margin < looseness_worst_margin - 1e-6 {
                        looseness_worst_margin = margin;
                        looseness_dumps_done += 1;
                        run_looseness_probe(graph, worst, engine, margin);
                    }
                }
            }

            let domains_to_process = prefilter_batch(
                batch,
                thresholds,
                conjunctive,
                self.config.max_depth,
                cuts_enabled,
                &mut lifecycle,
                &mut cut_pool,
                &mut clause_store,
            )?;

            if domains_to_process.is_empty() {
                continue;
            }

            // Default-off, verdict-inert NeuralSAT-style BCP shadow. Observe
            // only the deterministic frontier-first survivor and expose at
            // most one canonical exact literal. The replay runtime presents
            // its immutable graph/root/objective/property fingerprint and the
            // store compares it exactly before scanning. No history, phase,
            // bound, queue, scheduling decision, or verdict is mutated.
            if let (Some(replay), Some(domain)) =
                (clause_replay.as_ref(), domains_to_process.first())
            {
                if let Some(implication) =
                    replay.bcp_shadow_first_implication(&clause_store, &domain.history)
                {
                    tracing::info!(
                        depth = domain.depth(),
                        node_name = %implication.node_name,
                        neuron_idx = implication.neuron_idx,
                        forced_active = implication.is_active,
                        provenance = ?implication.provenance,
                        source_clause_len = implication.source_clause_len,
                        "BICCOS BCP shadow exact implication (not published)"
                    );
                }
            }

            if self.config.enable_clip_interm_domain && !bounded_shared_active {
                // This is the outer executable wave, matching αβ-CROWN's
                // total_round. Inner branch scoring and bound updates all read
                // the same immutable stamp.
                let bab_iteration = batch_index.saturating_add(1);
                let _ = self.complete_clip_root_bounds_cache.set_bab_iteration(
                    graph,
                    input,
                    bab_iteration,
                );
            }

            let batch_start = Instant::now();
            let batch_width = domains_to_process.len();
            let batch_plan = GraphDomainBatchPlan::for_multi_objective(
                batch_index,
                batch_width,
                batch_size,
                engine.is_some(),
                conjunctive,
            );

            if batch_plan.execution_mode() == GraphDomainBatchExecutionMode::SharedExecutor {
                debug_assert!(use_batched_gpu, "shared multi-objective batch plan drifted");
                let cut_pool_ref = if cuts_enabled && !cut_pool.is_empty() {
                    Some(&cut_pool)
                } else {
                    None
                };
                let domain_refs: Vec<_> = domains_to_process.iter().collect();
                // W is one-shot: it may score only the children split directly
                // from the root in executable shared batch zero. Taking it here
                // makes reuse by later waves/descendants structurally impossible.
                let selective_candidate_for_wave = if batch_index == 0 {
                    selective_root_alpha_candidate.take()
                } else {
                    None
                };

                // Wall clock for the WHOLE executable wave, not just the dense
                // stage `mo-wave-stage` already prints. On cifar100 only two
                // stages complete inside a ~42s BaB window, so most of the round
                // is spent outside that stage timer; this is the outer bracket
                // that says how much.
                let wave_round_probe =
                    crate::phase_telemetry::phase_telemetry_enabled().then(Instant::now);
                let results = GraphDomainBatchExecutor::execute_multi_objective(
                    self,
                    MultiObjectiveBatchRequest {
                        bab_round: batch_index,
                        graph,
                        domains: &domain_refs,
                        relu_nodes: &relu_nodes,
                        objectives,
                        thresholds,
                        engine: engine.ok_or_else(|| {
                            ny_core::NyError::InvalidSpec(
                                "shared multi-objective executor requires a GemmEngine".into(),
                            )
                        })?,
                        cut_pool: cut_pool_ref,
                        selective_root_alpha_candidate: selective_candidate_for_wave.as_ref(),
                    },
                );
                if let Some(t) = wave_round_probe {
                    eprintln!(
                        "[phase] mo-wave-round round={} domains={} secs={:.2}",
                        batch_index,
                        domain_refs.len(),
                        t.elapsed().as_secs_f64()
                    );
                }

                let queue_update_start = Instant::now();
                // Shadow-only BICCOS-Q source interception. Inspect the same
                // completed verified children offered to conflict-clause
                // replay, before queue folding consumes `results`. A typed
                // deadline in any sibling suppresses this optional tail. The
                // observer has no mutable solver-state handle.
                if let Some(stage0) = biccos_q_stage0.as_mut() {
                    let terminal_deadline = results.iter().any(|result| {
                        matches!(
                            result,
                            crate::beta_crown::engine::domain_results::
                                MultiObjectiveGraphDomainResult::DeadlineExpired
                        )
                    });
                    if !terminal_deadline {
                        let wave_children: Vec<_> = results
                            .iter()
                            .flat_map(|result| {
                                match result {
                                    crate::beta_crown::engine::domain_results::
                                        MultiObjectiveGraphDomainResult::Children(children)
                                    | crate::beta_crown::engine::domain_results::
                                        MultiObjectiveGraphDomainResult::ChildrenWithViolatedDrop(
                                            children,
                                        ) => children.as_slice(),
                                    _ => &[],
                                }
                            })
                            .collect();
                        let wave_histories: Vec<_> = wave_children
                            .iter()
                            .map(|(child, _)| &child.history)
                            .collect();
                        stage0.observe_wave(&wave_histories);
                        for (child, all_verified) in wave_children {
                            if *all_verified {
                                stage0.observe_verified_close(
                                    &child.history,
                                    &child.beta_state,
                                    &wave_histories,
                                );
                            }
                        }
                    }
                }
                // Only completed, sound child closes are replay sources. A
                // typed deadline in any sibling suppresses this optional tail
                // entirely so replay cannot consume the enclosing reserve.
                if let Some(replay) = clause_replay.as_mut() {
                    let terminal_deadline = results.iter().any(|result| {
                        matches!(
                            result,
                            crate::beta_crown::engine::domain_results::
                                MultiObjectiveGraphDomainResult::DeadlineExpired
                        )
                    });
                    if !terminal_deadline {
                        for result in &results {
                            let (children, stage1_provenance) = match result {
                                crate::beta_crown::engine::domain_results::
                                    MultiObjectiveGraphDomainResult::Children(children) => (
                                        children,
                                        crate::beta_crown::conflict_clause_replay::
                                            BiccosQStage1SourceProvenance::
                                                SharedMultiObjectiveChildren,
                                    ),
                                crate::beta_crown::engine::domain_results::
                                    MultiObjectiveGraphDomainResult::ChildrenWithViolatedDrop(
                                        children,
                                    ) => (
                                        children,
                                        crate::beta_crown::conflict_clause_replay::
                                            BiccosQStage1SourceProvenance::
                                                SharedMultiObjectiveChildrenWithViolatedSiblingDrop,
                                    ),
                                _ => continue,
                            };
                            for (child, all_verified) in children {
                                if *all_verified {
                                    // Stage 1 is subordinate to this already-armed
                                    // exact replay runtime. β/gradient values only
                                    // rank an opaque strict subset; acceptance
                                    // crosses the same replay/token/store boundary
                                    // as the established one-deletion policy.
                                    let stage1_accepted = replay
                                        .try_biccos_q_stage1_verified_close(
                                            self,
                                            graph,
                                            engine,
                                            &child.history,
                                            &child.beta_state,
                                            stage1_provenance,
                                            &mut clause_store,
                                        );
                                    if !stage1_accepted {
                                        replay.try_generalize_verified_close(
                                            self,
                                            graph,
                                            engine,
                                            &child.history,
                                            &mut clause_store,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                // Graph-MIP LEAF oracle context (increment 6): only built when
                // the default-on CLI attached an oracle; `None` under the
                // exact-zero kill switch keeps the requeue byte-identical.
                let leaf_ctx = (!bounded_shared_active)
                    .then(|| self.graph_mip_leaf_oracle())
                    .flatten()
                    .map(|oracle| LeafOracleCtx {
                        oracle,
                        graph,
                        objectives,
                        thresholds,
                        deadline: Some(bab_deadline),
                    });
                if let Some(result) = poll_bounded_shared_publication(
                    bounded_shared_active,
                    engine,
                    &mut lifecycle,
                    cut_pool.total_generated,
                )? {
                    return Ok(result);
                }
                // The fold itself: leaf-oracle consults, cut generation and the
                // heap pushes. `queue_update_start` above also covers the two
                // optional replay tails, so this narrower bracket is what
                // attributes the round's residue to the fold rather than to
                // them. `queue` is only pushed to here, never popped, so the
                // length delta IS the children enqueued.
                let queue_fold_probe = crate::phase_telemetry::phase_telemetry_enabled()
                    .then(|| (Instant::now(), queue.len()));
                let (batch_apply_status, leaf_violation) = apply_batched_results(
                    results,
                    &mut queue,
                    &mut cut_pool,
                    &mut lifecycle,
                    cuts_enabled,
                    leaf_ctx.as_ref(),
                    &mut clause_store,
                )?;
                if let Some((t, queued_before)) = queue_fold_probe {
                    eprintln!(
                        "[phase] mo-queue-fold round={} children_queued={} secs={:.2}",
                        batch_index,
                        queue.len().saturating_sub(queued_before),
                        t.elapsed().as_secs_f64()
                    );
                }
                // Graph-MIP LEAF sat return (#mip-leaf-witness). The oracle
                // produced an in-box point that a real forward pass through
                // THIS graph places inside the unsafe set of EVERY objective
                // row, so no continuation of the search can make the region
                // safe — the run is otherwise condemned to burn to
                // Unknown/timeout. Terminal, and checked BEFORE the
                // deadline/publication polls below: the candidate is only ever
                // produced by `consult_leaf_oracle`, which already REVOKES any
                // verdict reaching it at or after `bab_deadline`, so a latched
                // candidate is by construction pre-deadline evidence and is
                // strictly better than the Timeout that would replace it. The
                // queue is untouched (the child was requeued), and the CLI
                // re-confirms this witness through the unchanged
                // `gate_sat_with_trusted_oracle` before any `sat` is emitted.
                if let Some(status) = leaf_violation {
                    lifecycle.cuts_generated = cut_pool.total_generated;
                    return Ok(lifecycle.build_result(status));
                }
                if let Some(result) = poll_bounded_shared_publication(
                    bounded_shared_active,
                    engine,
                    &mut lifecycle,
                    cut_pool.total_generated,
                )? {
                    return Ok(result);
                }
                if let Some(result) = finish_shared_multi_objective_batch(
                    batch_apply_status,
                    &mut lifecycle,
                    cut_pool.total_generated,
                    || {
                        batch_plan.emit_to_sink(
                            self.graph_domain_batch_metrics_sink(),
                            GraphDomainBatchEmitTiming::new(batch_start.elapsed().as_secs_f64())
                                .with_queue_update(queue_update_start.elapsed().as_secs_f64()),
                        )
                    },
                )? {
                    return Ok(result);
                }
                if let Some(result) = poll_bounded_shared_publication(
                    bounded_shared_active,
                    engine,
                    &mut lifecycle,
                    cut_pool.total_generated,
                )? {
                    return Ok(result);
                }
                batch_index += 1;
                continue;
            }

            // Graph-MIP LEAF sat return channel for the sequential lane
            // (#mip-leaf-witness). This is the lane CONJUNCTIVE properties run
            // on (the batched lane above requires `!conjunctive`), and there
            // the "violates every objective row" test the oracle's witness must
            // pass is exactly the property's own violation predicate.
            let mut sequential_leaf_violation = None;
            let sequential_status =
                self.process_multi_objective_domains_sequential(SequentialMultiObjectiveContext {
                    graph,
                    domains_to_process,
                    relu_nodes: &relu_nodes,
                    objectives,
                    thresholds,
                    engine,
                    cut_pool: &mut cut_pool,
                    clause_store: &mut clause_store,
                    queue: &mut queue,
                    domains_verified: &mut lifecycle.domains_verified,
                    unresolved_due_to_no_branch: &mut lifecycle.unresolved_due_to_no_branch,
                    unresolved_due_to_violated_drop: &mut lifecycle.unresolved_due_to_violated_drop,
                    unresolved_due_to_propagation_failure: &mut lifecycle
                        .unresolved_due_to_propagation_failure,
                    leaf_violation: &mut sequential_leaf_violation,
                    conjunctive,
                    deadline: Some(bab_deadline),
                })?;
            batch_plan.emit_to_sink(
                self.graph_domain_batch_metrics_sink(),
                GraphDomainBatchEmitTiming::new(batch_start.elapsed().as_secs_f64()),
            )?;
            // Terminal, and ahead of the deadline check for the same reason as
            // the batched lane: `consult_leaf_oracle` revokes any verdict that
            // arrives at or after `bab_deadline`, so a latched candidate is
            // pre-deadline evidence that a concrete point violates the whole
            // property — strictly better than the Timeout it replaces. The
            // child that produced it is still on the queue, and the CLI's
            // unchanged `gate_sat_with_trusted_oracle` re-confirms the witness
            // with a real ONNX-Runtime forward before any `sat` is emitted.
            if let Some(status) = sequential_leaf_violation {
                lifecycle.cuts_generated = cut_pool.total_generated;
                return Ok(lifecycle.build_result(status));
            }
            if sequential_status == SequentialMultiObjectiveBatchStatus::DeadlineExpired {
                lifecycle.cuts_generated = cut_pool.total_generated;
                return Ok(lifecycle.timeout_result());
            }
            batch_index += 1;
        }

        lifecycle.cuts_generated = cut_pool.total_generated;
        if clause_store.is_enabled() {
            tracing::debug!(
                clause_pruned = clause_store.pruned_count(),
                replay_cross_pruned = clause_store.replay_pruned_count(),
                "Graph multi-objective BaB conflict-clause learning stats"
            );
        }
        if prof {
            eprintln!(
                "[NY_ACASXU_PROF] queue_drained: explored={} verified={} queue={} max_depth={} elapsed={:.2}s",
                lifecycle.domains_explored,
                lifecycle.domains_verified,
                queue.len(),
                lifecycle.max_depth_reached,
                lifecycle.start_time.elapsed().as_secs_f64(),
            );
        }
        Ok(finalize_multi_objective_result(
            &lifecycle,
            queue.is_empty(),
        ))
    }
}

/// NY_MO_BAB_TRACE (dark, diagnostic-only): the global worst still-*unverified*
/// objective lower bound across all open domains — i.e. the min over open domains
/// of the min over that domain's unverified objectives of the objective LB. This is
/// how far below its verify threshold the single hardest remaining objective sits;
/// as BaB progresses it should climb toward 0. Returns `None` when no open domain
/// carries any unverified objective (nothing left to close).
///
/// Pure reduction: no solver, no env, no I/O — split out so a unit test can pin it
/// on a synthetic domain set. Each item is one open domain as
/// `(objective_bounds, verified)`; a `verified` slice shorter than `bounds` (or an
/// index past its end) treats the missing entry as unverified (defense-in-depth —
/// a truncated flag vec must not hide an open objective).
pub(crate) fn worst_unverified_objective_lb<'a, I>(open_domains: I) -> Option<f32>
where
    I: IntoIterator<Item = (&'a [(f32, f32)], &'a [bool])>,
{
    let mut worst: Option<f32> = None;
    for (bounds, verified) in open_domains {
        for (i, (lo, _up)) in bounds.iter().enumerate() {
            if verified.get(i).copied().unwrap_or(false) {
                continue; // objective already verified in this domain — skip.
            }
            worst = Some(match worst {
                Some(w) => w.min(*lo),
                None => *lo,
            });
        }
    }
    worst
}

/// Format the one-line `[bab-trace]` progress record (NY_MO_BAB_TRACE). Split out
/// from the loop so a unit test can pin the exact greppable wire format without a
/// solver. An empty frontier is rendered independently from `worst_lb`: it can
/// mean either a clean proof drain or unresolved deadline-abandoned coverage,
/// as exposed by the adjacent `unresolved` field. For a non-empty frontier,
/// `worst_lb=None` means every queued domain is fully verified.
pub(crate) fn format_bab_trace_line(
    wall_elapsed_secs: f64,
    worst_lb: Option<f32>,
    open_domains: usize,
    max_depth: usize,
    domains_processed: usize,
    has_unresolved: bool,
) -> String {
    let worst = if open_domains == 0 {
        "empty-frontier".to_string()
    } else {
        match worst_lb {
            Some(w) => format!("{w:.6}"),
            None => "all-verified".to_string(),
        }
    };
    format!(
        "[bab-trace] t={wall_elapsed_secs:.3}s worst_unverified_lb={worst} \
         open_domains={open_domains} max_depth={max_depth} domains_processed={domains_processed} \
         unresolved={has_unresolved}"
    )
}

/// NY_LOOSENESS_PROBE (dark, diagnostic-only): for one worst subdomain, sample many
/// inputs in its box (respecting the domain's ReLU-split constraints via rejection),
/// forward the TRUE network (faithful center-collapse point forward), and dump for
/// EVERY intermediate node NY's computed [l,u] vs the TRUE sampled [min,max] plus the
/// per-node looseness (NY_width − true_width) and one-sided slack. Print-only; it never
/// mutates a bound or a verdict. Any error is swallowed (the probe must not perturb a run).
fn run_looseness_probe(
    graph: &GraphNetwork,
    worst: &crate::beta_crown::domain::MultiObjectiveGraphBabDomain,
    engine: Option<&dyn GemmEngine>,
    margin: f32,
) {
    // Per-sample forwards are batch-1: the GPU dispatch overhead per node dominates,
    // so default to the CPU sound forward (faithful, far faster at batch-1). Set
    // NY_LOOSENESS_GPU=1 to force the GPU engine instead.
    let fwd_engine = if std::env::var("NY_LOOSENESS_GPU").ok().as_deref() == Some("1") {
        engine
    } else {
        None
    };
    use crate::layers::Layer;
    use ny_tensor::BoundedTensor;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::collections::HashMap;

    let n_samples = std::env::var("NY_LOOSENESS_SAMPLES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2000);
    let seed = std::env::var("NY_LOOSENESS_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0xC1FA_1498);

    let input_box = worst.input_bounds();
    let lower = input_box.lower();
    let upper = input_box.upper();
    let constraints = &worst.history().constraints;

    eprintln!(
        "[looseness] START depth={} margin(min-lower)={:.6} n_split_constraints={} n_samples={} seed={}",
        worst.depth(),
        margin,
        constraints.len(),
        n_samples,
        seed
    );

    // Precompute (pre-activation-node-name, neuron_idx, is_active) for each split.
    // A hidden-ReLU split constrains its INPUT node (the pre-activation), not the ReLU
    // output. NETWORK_INPUT (or missing) pre-activations are un-evaluable here → skipped.
    let mut split_checks: Vec<(String, usize, bool)> = Vec::new();
    for c in constraints {
        if let Some(node) = graph.node(&c.node_name) {
            if let Some(inp) = node.inputs().first() {
                split_checks.push((inp.clone(), c.neuron_idx, c.is_active));
            }
        }
    }

    // The hidden-ReLU splits define a thin sub-region of the eps-box that uniform
    // sampling almost never hits (all 6 satisfied simultaneously is rare), so STRICT
    // rejection typically yields 0 accepted samples. By default we therefore DO NOT
    // reject: the full-box true range is EXACT for nodes upstream of the first split
    // (splits are downstream and cannot restrict them) — precisely where the task
    // hypothesizes the looseness first enters — and a (too-wide) superset for
    // downstream nodes, so their reported looseness is a conservative LOWER BOUND.
    // Set NY_LOOSENESS_REJECT=1 to enforce strict split-rejection instead.
    let strict_reject = std::env::var("NY_LOOSENESS_REJECT").ok().as_deref() == Some("1");
    let mut rng = StdRng::seed_from_u64(seed);
    // Per-node running true min/max across accepted samples (flat neuron order).
    let mut true_min: HashMap<String, Vec<f32>> = HashMap::new();
    let mut true_max: HashMap<String, Vec<f32>> = HashMap::new();
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    // Split-feasibility counter (how many samples satisfy ALL splits), reported even
    // when strict_reject is off so we can see how thin the split region is.
    let mut split_feasible = 0usize;

    let probe_start = Instant::now();
    for si in 0..n_samples {
        if si > 0 && si % 250 == 0 {
            eprintln!(
                "[looseness] progress: {si}/{n_samples} accepted={accepted} rejected={rejected} elapsed={:.1}s",
                probe_start.elapsed().as_secs_f64()
            );
        }
        let mut point = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(input_box.shape()));
        for (v, (l, u)) in point.iter_mut().zip(lower.iter().zip(upper.iter())) {
            *v = if u > l { rng.random_range(*l..=*u) } else { *l };
        }
        let pt = match BoundedTensor::concrete(point) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let cache = match graph.collect_node_activations_pointwise(&pt, fwd_engine) {
            Ok(c) => c,
            Err(_) => {
                rejected += 1;
                continue;
            }
        };

        // Evaluate every hidden-ReLU split half-space on the true pre-activation.
        let mut splits_ok = true;
        for (node_name, idx, is_active) in &split_checks {
            if let Some(bt) = cache.get(node_name) {
                let center = bt.center();
                if let Some(val) = center.iter().nth(*idx).copied() {
                    // is_active ⇒ preact ≥ 0 ; !is_active ⇒ preact ≤ 0
                    if (*is_active && val < 0.0) || (!*is_active && val > 0.0) {
                        splits_ok = false;
                        break;
                    }
                }
            }
        }
        if splits_ok {
            split_feasible += 1;
        }
        if strict_reject && !splits_ok {
            rejected += 1;
            continue;
        }
        accepted += 1;

        for (name, bt) in &cache {
            let center = bt.center();
            let mn = true_min
                .entry(name.clone())
                .or_insert_with(|| vec![f32::INFINITY; center.len()]);
            for (slot, v) in mn.iter_mut().zip(center.iter()) {
                if *v < *slot {
                    *slot = *v;
                }
            }
            let mx = true_max
                .entry(name.clone())
                .or_insert_with(|| vec![f32::NEG_INFINITY; center.len()]);
            for (slot, v) in mx.iter_mut().zip(center.iter()) {
                if *v > *slot {
                    *slot = *v;
                }
            }
        }
    }

    eprintln!(
        "[looseness] sampling done: accepted={accepted} rejected={rejected} strict_reject={strict_reject} split_feasible={split_feasible}/{n_samples} (fraction of box satisfying all {} splits)",
        split_checks.len()
    );
    if accepted == 0 {
        eprintln!("[looseness] NO accepted samples (splits may be infeasible in the box) — cannot compare");
        return;
    }

    // Build the per-node comparison table.
    struct Row {
        name: String,
        is_relu: bool,
        ny_l: f32,
        ny_u: f32,
        t_min: f32,
        t_max: f32,
        node_looseness: f32,   // NY node-width − true node-width
        max_neuron_loose: f32, // worst single-neuron (NY_width − true_width)
        sum_neuron_loose: f64, // total slack contributed by this node
        lower_slack: f32,      // true_min − NY_l   (≥0 means NY_l is below true_min)
        upper_slack: f32,      // NY_u − true_max
    }

    let mut rows: Vec<Row> = Vec::new();
    let order = graph.exec_order().ok();
    let node_iter: Vec<String> = match &order {
        Some(o) => o.to_vec(),
        None => worst.node_bounds().keys().cloned().collect(),
    };

    for name in node_iter {
        let Some(bt) = worst.node_bounds().get(&name) else {
            continue;
        };
        let (Some(tmn), Some(tmx)) = (true_min.get(&name), true_max.get(&name)) else {
            continue;
        };
        let ny_lo = bt.lower();
        let ny_hi = bt.upper();
        if ny_lo.len() != tmn.len() {
            // shape mismatch (unexpected) — skip rather than mis-align neurons.
            continue;
        }
        let ny_l = ny_lo.iter().copied().fold(f32::INFINITY, f32::min);
        let ny_u = ny_hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let t_min = tmn.iter().copied().fold(f32::INFINITY, f32::min);
        let t_max = tmx.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut max_neuron_loose = f32::NEG_INFINITY;
        let mut sum_neuron_loose = 0.0f64;
        for i in 0..tmn.len() {
            let ny_w = ny_hi.iter().nth(i).copied().unwrap_or(0.0)
                - ny_lo.iter().nth(i).copied().unwrap_or(0.0);
            let t_w = tmx[i] - tmn[i];
            let loose = ny_w - t_w;
            if loose > max_neuron_loose {
                max_neuron_loose = loose;
            }
            sum_neuron_loose += loose.max(0.0) as f64;
        }

        let is_relu = graph
            .node(&name)
            .map(|n| matches!(n.layer(), Layer::ReLU(_)))
            .unwrap_or(false);

        rows.push(Row {
            name,
            is_relu,
            ny_l,
            ny_u,
            t_min,
            t_max,
            node_looseness: (ny_u - ny_l) - (t_max - t_min),
            max_neuron_loose,
            sum_neuron_loose,
            lower_slack: t_min - ny_l,
            upper_slack: ny_u - t_max,
        });
    }

    // Sort by summed per-neuron looseness (total slack the layer carries), desc.
    rows.sort_by(|a, b| {
        b.sum_neuron_loose
            .partial_cmp(&a.sum_neuron_loose)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    eprintln!("[looseness] ===== PER-NODE LOOSENESS TABLE (sorted by summed per-neuron slack, desc) =====");
    eprintln!(
        "[looseness] {:<12} {:>5} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>12} {:>11} {:>11}",
        "node",
        "relu",
        "NY_l",
        "NY_u",
        "true_min",
        "true_max",
        "node_loose",
        "max_nloose",
        "sum_nloose",
        "lo_slack",
        "up_slack"
    );
    for r in &rows {
        eprintln!(
            "[looseness] {:<12} {:>5} {:>11.5} {:>11.5} {:>11.5} {:>11.5} {:>11.5} {:>11.5} {:>12.3} {:>11.5} {:>11.5}",
            r.name,
            if r.is_relu { "R" } else { "-" },
            r.ny_l,
            r.ny_u,
            r.t_min,
            r.t_max,
            r.node_looseness,
            r.max_neuron_loose,
            r.sum_neuron_loose,
            r.lower_slack,
            r.upper_slack,
        );
    }
    eprintln!("[looseness] ===== END TABLE =====");
}

#[cfg(test)]
mod post_root_engine_handoff_tests {
    use std::cell::Cell;
    use std::sync::Arc;
    use std::time::Duration;

    use ndarray::{arr1, arr2};
    use ny_core::{GpuCrownBackward, GpuCrownLayer, GpuCrownResult, NaiveCpuGemmEngine, NyError};

    use super::*;
    use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
    use crate::beta_crown::engine::domain_results::MultiObjectiveGraphDomainResult;
    use crate::{GraphNode, LinearLayer};

    struct MockHandoffEngine {
        deadline_safe_bab_surface: bool,
        sound: bool,
        cooperative_deadline: bool,
    }

    impl MockHandoffEngine {
        fn new(deadline_safe_bab_surface: bool, sound: bool, cooperative_deadline: bool) -> Self {
            Self {
                deadline_safe_bab_surface,
                sound,
                cooperative_deadline,
            }
        }
    }

    impl GemmEngine for MockHandoffEngine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }

        fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
            Some(self)
        }

        fn supports_deadline_safe_post_root_multi_objective_bab(&self) -> bool {
            self.deadline_safe_bab_surface
        }
    }

    impl GpuCrownBackward for MockHandoffEngine {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp("test engine".into()))
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            self.sound
        }

        fn honors_crown_backward_deadline(&self) -> bool {
            self.cooperative_deadline
        }
    }

    #[test]
    fn exact_env_override_inherits_only_when_absent() {
        assert!(!resolve_mo_cuda_factory_engine_handoff_gate(false, None));
        assert!(resolve_mo_cuda_factory_engine_handoff_gate(true, None));

        for typed in [false, true] {
            assert!(resolve_mo_cuda_factory_engine_handoff_gate(
                typed,
                Some(OsStr::new("1"))
            ));
            for raw in ["", "0", "01", "true", " 1", "1 "] {
                assert!(
                    !resolve_mo_cuda_factory_engine_handoff_gate(typed, Some(OsStr::new(raw))),
                    "present runtime spelling {raw:?} must force the handoff off"
                );
            }
        }
    }

    #[test]
    fn caller_engine_has_precedence_without_observing_factory_slot() {
        let caller = NaiveCpuGemmEngine;
        let factory_lookups = Cell::new(0usize);
        let deadline = Instant::now() + Duration::from_secs(1);

        let selected = resolve_post_root_multi_objective_engine(
            Some(&caller),
            true,
            deadline,
            Instant::now,
            || {
                factory_lookups.set(factory_lookups.get() + 1);
                None
            },
        )
        .expect("caller engine must be retained");

        assert!(std::ptr::eq(selected, &caller as &dyn GemmEngine));
        assert_eq!(
            factory_lookups.get(),
            0,
            "caller precedence must avoid even a get-only factory-slot observation"
        );
    }

    #[test]
    fn dark_or_expired_handoff_does_not_observe_factory_slot() {
        let lookups = Cell::new(0usize);
        let live_deadline = Instant::now() + Duration::from_secs(1);
        let dark = resolve_post_root_multi_objective_engine(
            None,
            false,
            live_deadline,
            || panic!("dark handoff must not read the clock"),
            || {
                lookups.set(lookups.get() + 1);
                None
            },
        );
        assert!(dark.is_none());

        let expired_at = Instant::now();
        let expired = resolve_post_root_multi_objective_engine(
            None,
            true,
            expired_at,
            || expired_at,
            || {
                lookups.set(lookups.get() + 1);
                None
            },
        );
        assert!(expired.is_none());
        assert_eq!(lookups.get(), 0);
    }

    #[test]
    fn factory_candidate_requires_deadline_safe_sound_cooperative_surface() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(1);

        for (deadline_safe_bab_surface, sound, cooperative_deadline, expected) in [
            (false, false, false, false),
            (false, false, true, false),
            (false, true, false, false),
            (false, true, true, false),
            (true, false, false, false),
            (true, false, true, false),
            (true, true, false, false),
            (true, true, true, true),
        ] {
            let candidate =
                MockHandoffEngine::new(deadline_safe_bab_surface, sound, cooperative_deadline);
            let lookups = Cell::new(0usize);
            let selected = resolve_post_root_multi_objective_engine(
                None,
                true,
                deadline,
                || now,
                || {
                    lookups.set(lookups.get() + 1);
                    Some(&candidate)
                },
            );

            assert_eq!(
                selected.is_some(),
                expected,
                "deadline_safe_bab_surface={deadline_safe_bab_surface} sound={sound} \
                 cooperative_deadline={cooperative_deadline}"
            );
            assert_eq!(lookups.get(), 1, "the slot must be resolved exactly once");
        }
    }

    #[test]
    fn eligible_handoff_selects_and_executes_first_shared_wave() {
        let candidate = MockHandoffEngine::new(true, true, true);
        let now = Instant::now();
        let selected = resolve_post_root_multi_objective_engine(
            None,
            true,
            now + Duration::from_secs(1),
            || now,
            || Some(&candidate),
        );
        let plan = GraphDomainBatchPlan::for_multi_objective(0, 2, 2, selected.is_some(), false);

        assert_eq!(
            plan.execution_mode(),
            GraphDomainBatchExecutionMode::SharedExecutor
        );

        let linear = LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32])))
            .expect("single-output linear layer");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
        graph.set_output("linear");
        let input = Arc::new(
            BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn())
                .expect("finite input"),
        );
        let node_bounds = graph.collect_node_bounds(&input).expect("root node bounds");
        let domain = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(0.0, 1.0)],
            input.as_ref(),
            &[0.5],
            false,
        )
        .expect("root multi-objective domain");
        let verifier = BetaCrownVerifier::new(crate::beta_crown::BetaCrownConfig::default());
        let results = GraphDomainBatchExecutor::execute_multi_objective(
            &verifier,
            MultiObjectiveBatchRequest {
                bab_round: 0,
                graph: &graph,
                domains: &[&domain],
                relu_nodes: &[],
                objectives: &[vec![1.0]],
                thresholds: &[0.5],
                engine: selected.expect("eligible handoff"),
                cut_pool: None,
                selective_root_alpha_candidate: None,
            },
        );

        assert!(
            matches!(
                results.as_slice(),
                [MultiObjectiveGraphDomainResult::NoUnstable {
                    all_verified: true,
                    any_violated: false,
                }]
            ),
            "first shared wave must execute successfully, got {results:?}"
        );
    }
}

#[cfg(test)]
mod shared_batch_deadline_tests {
    use std::collections::BinaryHeap;
    use std::time::Instant;

    use ny_core::NyError;

    use super::*;
    use crate::beta_crown::bab_cuts::GraphCutPool;
    use crate::beta_crown::conflict_clauses_graph::GraphClauseStore;
    use crate::beta_crown::engine::domain_results::MultiObjectiveGraphDomainResult;
    use crate::beta_crown::result::BabVerificationStatus;

    struct ExpiredPublicationEngine;

    impl GemmEngine for ExpiredPublicationEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            unreachable!("publication test must not enter GEMM")
        }

        fn poll_crown_backward_deadline(&self) -> Result<()> {
            Err(NyError::DeadlineExceeded(
                "injected bounded publication expiry".into(),
            ))
        }

        fn forbids_unbounded_cpu_fallback(&self) -> bool {
            true
        }
    }

    #[test]
    fn bounded_executor_disables_post_root_cuts() {
        assert!(!post_root_cuts_enabled(true, true));
        assert!(!post_root_cuts_enabled(false, true));
        assert!(post_root_cuts_enabled(true, false));
        assert!(!post_root_cuts_enabled(false, false));
    }

    #[test]
    fn typed_deadline_returns_timeout_without_calling_fallible_metrics_sink() {
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut clause_store = GraphClauseStore::disabled();
        let (status, _leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::DeadlineExpired],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            None,
            &mut clause_store,
        )
        .expect("typed producer result must fold");
        let mut sink_called = false;

        let result = finish_shared_multi_objective_batch(status, &mut lifecycle, 17, || {
            sink_called = true;
            Err(NyError::InternalError(
                "terminal metrics sink must not run".to_string(),
            ))
        })
        .expect("typed deadline must not expose the sink error")
        .expect("typed deadline must return a terminal result");

        assert!(!sink_called);
        assert_eq!(result.result, BabVerificationStatus::Timeout);
        assert_eq!(result.cuts_generated, 17);
    }

    #[test]
    fn completed_batch_preserves_fallible_metrics_contract() {
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let error = finish_shared_multi_objective_batch(
            MultiObjectiveBatchApplyStatus::Completed,
            &mut lifecycle,
            0,
            || {
                Err(NyError::InternalError(
                    "completed metrics sink failure".to_string(),
                ))
            },
        )
        .expect_err("completed batches must still propagate metrics sink failure");

        assert!(matches!(error, NyError::InternalError(_)));
    }

    #[test]
    fn bounded_post_queue_publication_expiry_forces_timeout() {
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let result = poll_bounded_shared_publication(
            true,
            Some(&ExpiredPublicationEngine),
            &mut lifecycle,
            23,
        )
        .expect("typed publication deadline must be graceful")
        .expect("expired bounded publication must be terminal");
        assert_eq!(result.result, BabVerificationStatus::Timeout);
        assert_eq!(result.cuts_generated, 23);
    }
}

#[cfg(test)]
mod bab_trace_tests {
    use super::{format_bab_trace_line, worst_unverified_objective_lb};

    /// The min-reduction picks the globally lowest LB across *unverified*
    /// objectives of *all* open domains, ignoring verified ones.
    #[test]
    fn worst_unverified_lb_min_over_open_and_unverified() {
        // Domain A: obj0 verified (LB 5.0, ignored), obj1 unverified LB -0.30.
        let a_bounds = [(5.0f32, 6.0), (-0.30, 0.10)];
        let a_verified = [true, false];
        // Domain B: obj0 unverified LB -0.80 (the global worst), obj1 unverified LB 0.20.
        let b_bounds = [(-0.80f32, 0.05), (0.20, 0.40)];
        let b_verified = [false, false];

        let worst = worst_unverified_objective_lb([
            (a_bounds.as_slice(), a_verified.as_slice()),
            (b_bounds.as_slice(), b_verified.as_slice()),
        ]);
        // -0.80 (B.obj0) beats -0.30 (A.obj1); A.obj0's 5.0 is verified -> ignored.
        assert_eq!(worst, Some(-0.80));
    }

    /// When every open domain has all objectives verified, there is nothing left
    /// to close -> `None`. Same for an empty domain set.
    #[test]
    fn worst_unverified_lb_none_when_all_verified_or_empty() {
        let bounds = [(0.10f32, 0.20), (0.30, 0.40)];
        let verified = [true, true];
        let all_verified =
            worst_unverified_objective_lb([(bounds.as_slice(), verified.as_slice())]);
        assert_eq!(all_verified, None);

        let empty: [(&[(f32, f32)], &[bool]); 0] = [];
        assert_eq!(worst_unverified_objective_lb(empty), None);
    }

    /// Defense-in-depth: a `verified` slice shorter than `bounds` treats the
    /// missing flag as unverified (a truncated vec must not hide an open objective).
    #[test]
    fn worst_unverified_lb_short_verified_counts_missing_as_open() {
        let bounds = [(0.50f32, 0.60), (-0.10, 0.10)];
        let verified = [true]; // obj1's flag missing -> treated as unverified.
        let worst = worst_unverified_objective_lb([(bounds.as_slice(), verified.as_slice())]);
        assert_eq!(worst, Some(-0.10));
    }

    /// Pin the exact greppable one-line wire format (stable `[bab-trace]` prefix).
    #[test]
    fn trace_line_format_is_stable() {
        let line = format_bab_trace_line(12.5, Some(-0.375), 42, 7, 1234, false);
        assert_eq!(
            line,
            "[bab-trace] t=12.500s worst_unverified_lb=-0.375000 \
             open_domains=42 max_depth=7 domains_processed=1234 unresolved=false"
        );
    }

    /// An empty heap is distinct from proof completion because the lifecycle
    /// may still carry unresolved work from the just-finished batch.
    #[test]
    fn trace_line_empty_frontier_exposes_unresolved_state() {
        let line = format_bab_trace_line(0.0, None, 0, 3, 99, true);
        assert_eq!(
            line,
            "[bab-trace] t=0.000s worst_unverified_lb=empty-frontier \
             open_domains=0 max_depth=3 domains_processed=99 unresolved=true"
        );
    }

    /// A non-empty queue whose domains have no unverified objective retains the
    /// historical `all-verified` spelling.
    #[test]
    fn trace_line_all_verified_queued_domains() {
        let line = format_bab_trace_line(0.0, None, 3, 3, 99, false);
        assert_eq!(
            line,
            "[bab-trace] t=0.000s worst_unverified_lb=all-verified \
             open_domains=3 max_depth=3 domains_processed=99 unresolved=false"
        );
    }
}
