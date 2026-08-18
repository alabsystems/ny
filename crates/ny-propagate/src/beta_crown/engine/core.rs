// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core β-CROWN verification entrypoints.

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{info, instrument};

use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::domain::BabDomain;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::{GraphNetwork, Network};

use super::cut_gate::CutGateState;
use super::verify_phases::{
    pop_domain_batch, record_cut_gate_batch, BabLoopState, InitialPhaseOutcome,
};
use super::{
    BetaCrownVerifier, GraphDomainBatchMetricsSink, InputSplitMetricsSink, ResnetSkeletonCache,
};

impl BetaCrownVerifier {
    fn new_with_resources(
        config: BetaCrownConfig,
        engine: Option<Arc<dyn GemmEngine>>,
        input_split_metrics_sink: Option<Arc<dyn InputSplitMetricsSink>>,
        graph_domain_batch_metrics_sink: Option<Arc<dyn GraphDomainBatchMetricsSink>>,
    ) -> Self {
        let mut config = config;
        // `BetaCrownConfig::timeout` is the verifier's concrete engine budget:
        // zero therefore means an immediately expired verifier. The CLI maps
        // its user-facing zero/unbounded sentinel to a representable long
        // engine horizon before construction. Use checked arithmetic so a
        // direct caller's platform-unrepresentable duration cannot panic.
        let now = Instant::now();
        // CONSEQUENCE WORTH KNOWING (documented 2026-08-17, behavior unchanged):
        // this is unconditional, so EVERY engine-driven run carries
        // `alpha_config.deadline == Some(..)`. That silently disables
        // `GradientMethod::AnalyticChain` on the DAG lane, which refuses under
        // `deadline.is_some()` (propagate_dag/gradients/mod.rs, #chain-grad gate)
        // — and the chain pass is the ONLY route to the `#envelope-grad` rule on
        // that lane. So arming NY_ALPHA_ENVELOPE_GRAD on a DAG graph through any
        // BetaCrownVerifier is INERT: the flag reads as set, no envelope code
        // runs, and the null looks like a measured negative.
        //
        // This landed AFTER every CPU-envelope measurement in the tree, so those
        // numbers are not reproducible on the current binary through that lane.
        // The fix is to make the replay cooperative (a private finite sub-budget
        // with all-or-nothing consumption), not to weaken the deadline here — an
        // engine without a deadline is the actual defect.
        config.alpha_config.deadline = Some(now.checked_add(config.timeout).unwrap_or(now));
        Self {
            config,
            engine,
            input_split_metrics_sink,
            graph_domain_batch_metrics_sink,
            joint_margin_closer: None,
            graph_mip_leaf_oracle: None,
            disjunctive_restart_root_cache: None,
            // #extract-skeleton increment 3: fresh (empty) per verifier —
            // entries are built lazily by the prep call sites and re-validated
            // against the current graph on every hit.
            skeleton_cache: ResnetSkeletonCache::default(),
            complete_clip_root_bounds_cache: Default::default(),
            complete_clip_deadline_overrides: Default::default(),
            gather_score_cache: Default::default(),
            adaptive_depth_shadow_fired: std::sync::atomic::AtomicBool::new(false),
            kfsb_f64_shadow_fired: std::sync::atomic::AtomicBool::new(false),
            attribution_diag_fired: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Create a new β-CROWN verifier.
    ///
    /// Sets `alpha_config.deadline` from `timeout` so that per-domain beta
    /// optimization loops can bail early when the verification budget is
    /// exhausted. Without this, `past_deadline()` always returns `false`
    /// and a single BaB batch can exceed the total budget by 2-10x. (#3109)
    pub fn new(config: BetaCrownConfig) -> Self {
        Self::new_with_resources(config, None, None, None)
    }

    /// Create a β-CROWN verifier with a stored GemmEngine.
    pub fn new_with_engine(config: BetaCrownConfig, engine: Arc<dyn GemmEngine>) -> Self {
        Self::new_with_resources(config, Some(engine), None, None)
    }

    /// Attach a runtime-only input-split metrics sink.
    #[must_use]
    pub fn with_input_split_metrics_sink(mut self, sink: Arc<dyn InputSplitMetricsSink>) -> Self {
        self.input_split_metrics_sink = Some(sink);
        self
    }

    /// Attach a runtime-only graph domain-batch metrics sink.
    #[must_use]
    pub fn with_graph_domain_batch_metrics_sink(
        mut self,
        sink: Arc<dyn GraphDomainBatchMetricsSink>,
    ) -> Self {
        self.graph_domain_batch_metrics_sink = Some(sink);
        self
    }

    /// Create a new β-CROWN verifier with a different config but reusing
    /// the engine from an existing verifier. This ensures sub-verifiers
    /// (per-constraint, reduced, bounds precomputation) inherit GPU
    /// acceleration without per-call engine threading (#3627).
    pub fn with_config_from(&self, config: BetaCrownConfig) -> Self {
        let mut v = Self::new_with_resources(
            config,
            self.engine_arc(),
            self.input_split_metrics_sink_arc(),
            self.graph_domain_batch_metrics_sink_arc(),
        );
        v.joint_margin_closer = self.joint_margin_closer.clone();
        v.graph_mip_leaf_oracle = self.graph_mip_leaf_oracle.clone();
        v.disjunctive_restart_root_cache = self.disjunctive_restart_root_cache.clone();
        v
    }

    /// Effective deadline for graph-BaB work in the current call.
    ///
    /// `alpha_config.deadline` is anchored when the verifier is constructed.
    /// A caller-supplied graph-BaB deadline can be earlier because the CLI
    /// ledger has already reserved time for post-BaB phases. Graph-BaB entry
    /// points install that earlier boundary in the shared override scope; every
    /// nested branch-selection, propagation, and Complete Clipping path must
    /// observe the minimum of the two.
    pub(crate) fn effective_graph_bab_deadline(&self) -> Option<Instant> {
        self.complete_clip_deadline_overrides
            .effective(self.config.alpha_config.deadline)
    }

    /// Whether the effective graph-BaB deadline has expired.
    pub(crate) fn past_effective_graph_bab_deadline(&self) -> bool {
        self.effective_graph_bab_deadline()
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    /// Attach a fresh, call-local exact root-map cache for deterministic
    /// grouped-disjunctive restarts. The cache remains bound to this exact
    /// packed property and exact original absolute deadline.
    #[must_use]
    pub fn with_fresh_disjunctive_restart_root_cache(
        mut self,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        clause_sizes: &[usize],
        overall_deadline: Option<Instant>,
    ) -> Self {
        let spec = super::graph::input_split::root_bounds::disjunctive_spec_identity(
            objectives,
            thresholds,
            clause_sizes,
        );
        self.disjunctive_restart_root_cache = Some(Arc::new(
            super::graph::input_split::root_bounds::InputSplitRootBoundsCache::new(
                spec,
                overall_deadline,
            ),
        ));
        self
    }

    /// Return the restart cache only when both the original absolute deadline
    /// and the exact packed property identity still match. Any mismatch fails
    /// closed to ordinary root collection.
    pub(crate) fn disjunctive_restart_root_cache(
        &self,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        clause_sizes: &[usize],
        overall_deadline: Option<Instant>,
    ) -> Option<&super::graph::input_split::root_bounds::InputSplitRootBoundsCache> {
        let cache = self.disjunctive_restart_root_cache.as_deref()?;
        if !cache.deadline_matches(overall_deadline) {
            return None;
        }
        let spec = super::graph::input_split::root_bounds::disjunctive_spec_identity(
            objectives,
            thresholds,
            clause_sizes,
        );
        cache.spec_matches(&spec).then_some(cache)
    }

    /// Attach a Graph-MIP LEAF oracle (increment 6,
    /// `docs/GRAPH_MIP_LEAF_SOLVER.md`): the graph ReLU-split BaB consults it
    /// right before requeueing an UNDECIDED child, so tractable subdomains
    /// (split premises pinned, few free binaries) can be decided exactly by
    /// the external MIP path. Runtime-only; `None` (never attached) is
    /// byte-for-byte inert. Preserved across `with_config_from` like the
    /// metrics sinks and the joint-margin closer.
    #[must_use]
    pub fn with_graph_mip_leaf_oracle(
        mut self,
        oracle: Arc<dyn crate::beta_crown::graph_mip_leaf::GraphMipLeafOracle>,
    ) -> Self {
        self.graph_mip_leaf_oracle = Some(oracle);
        self
    }

    /// The attached Graph-MIP leaf oracle, if any (see
    /// [`with_graph_mip_leaf_oracle`](Self::with_graph_mip_leaf_oracle)).
    pub(crate) fn graph_mip_leaf_oracle(
        &self,
    ) -> Option<&dyn crate::beta_crown::graph_mip_leaf::GraphMipLeafOracle> {
        self.graph_mip_leaf_oracle.as_deref()
    }

    /// Attach a per-domain joint-margin closer for same-LHS conjunctive
    /// (max-diff) input-split BaB (acasxu prop_2/3/4). The closer certifies a
    /// tighter JOINT lower bound than CROWN's single-conjunct MaxPool relaxation
    /// on the "hard shell" of near-boundary boxes where the plain input-split
    /// BaB diverges. Runtime-only; sound (only ever raises a domain's lower
    /// bound). See `joint_margin::JointMarginCloser`.
    #[must_use]
    pub fn with_joint_margin_closer(mut self, closer: Arc<super::JointMarginCloser>) -> Self {
        self.joint_margin_closer = Some(closer);
        self
    }

    pub(crate) fn joint_margin_closer(&self) -> Option<&super::JointMarginCloser> {
        self.joint_margin_closer.as_deref()
    }

    pub(crate) fn engine(&self) -> Option<&dyn GemmEngine> {
        self.engine.as_deref()
    }

    pub fn engine_arc(&self) -> Option<Arc<dyn GemmEngine>> {
        self.engine.clone()
    }

    pub(crate) fn input_split_metrics_sink(&self) -> Option<&dyn InputSplitMetricsSink> {
        self.input_split_metrics_sink.as_deref()
    }

    pub fn input_split_metrics_sink_arc(&self) -> Option<Arc<dyn InputSplitMetricsSink>> {
        self.input_split_metrics_sink.clone()
    }

    pub(crate) fn graph_domain_batch_metrics_sink(
        &self,
    ) -> Option<&dyn GraphDomainBatchMetricsSink> {
        self.graph_domain_batch_metrics_sink.as_deref()
    }

    pub fn graph_domain_batch_metrics_sink_arc(
        &self,
    ) -> Option<Arc<dyn GraphDomainBatchMetricsSink>> {
        self.graph_domain_batch_metrics_sink.clone()
    }

    pub(crate) fn resolve_engine<'a>(
        &'a self,
        engine: Option<&'a dyn GemmEngine>,
    ) -> Option<&'a dyn GemmEngine> {
        engine.or_else(|| self.engine())
    }

    /// Clone a graph and apply the verifier's conv-mode policy to graph-side CROWN.
    pub(crate) fn configured_graph_for_crown(&self, graph: &GraphNetwork) -> GraphNetwork {
        let mut configured = graph.clone();
        configured.set_use_patches_mode(self.config.use_patches());
        // Preset-scoped per-node CROWN-IBP time budget (#cgan-bn11-budget).
        // All-None (every preset that doesn't set the knobs) is byte-identical
        // to the historical constants.
        configured.set_crown_ibp_per_node_time_budget(self.config.crown_ibp_per_node_time_budget());
        configured.set_forward_linear_deadline_fallback_to_ibp(
            self.config
                .alpha_config
                .forward_linear_deadline_fallback_to_ibp,
        );
        // Carry the source's certified forward-linear reference map into the
        // clone (#w5-bab-throughput): `Clone` resets it and `set_use_patches_mode`
        // invalidates it, so every verify entry repaid the full O(L) certified
        // pass (~25s on cifar100) for a map already computed upstream (e.g. the
        // CLI-level graph warmed during the attack phase). Sound: the map depends
        // only on (structure + weights, input key), both identical here;
        // `use_patches_mode` is CROWN-backward-only and never read by the
        // forward-linear collection.
        configured.adopt_forward_linear_cache_from(graph);
        // Carry the source's input-keyed CROWN-IBP collection too
        // (#cgan-collection-cache): the disjunctive precheck's COMPLETE
        // root-box collection must reach the alpha warmup / BaB bootstrap on
        // this configured clone instead of being recomputed (truncated) under
        // the warmup's phase budget. Sound: same weights; the cache key
        // includes `use_patches_mode`, so if the conv-mode stamp above changed
        // it, the adopted entry simply misses.
        configured.adopt_crown_ibp_collection_cache_from(graph);
        // One fresh lock-free diagnostic stream per top-level configured graph.
        // Replacing the clone's Arc (rather than zeroing shared atomics) keeps
        // concurrent verifier calls independent; all later BaB domain clones
        // share this newly installed scope.
        configured.begin_crown_degradation_log_scope();
        configured
    }

    /// Verify with optional GPU acceleration via GemmEngine.
    ///
    /// Same as `verify`, but accepts an optional GemmEngine for GPU-accelerated
    /// linear layer CROWN backward passes.
    ///
    /// `deadline`: If `Some`, the BaB engine derives its phase budgets from
    /// remaining wall-clock time instead of `self.config.timeout` (#4321).
    pub fn verify_with_engine(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        let engine = self.resolve_engine(engine);
        self.verify_impl(network, input, threshold, engine, deadline)
    }

    /// Verify that output > threshold for all inputs in the bounded region.
    ///
    /// Returns Verified if we can prove output > threshold for all inputs,
    /// PotentialViolation if we find a region where output might be < threshold,
    /// Unknown if we can't determine (timeout/domain limit).
    ///
    /// # REQUIRES
    /// - `network` must have compatible layer dimensions (valid network)
    /// - `input` shape must match network's expected input dimension
    /// - `input.lower()[i] <= input.upper()[i]` for all elements (well-formed bounds)
    /// - `threshold` should be finite
    ///
    /// # ENSURES
    /// - If `Verified`: for all `x` in input region, `network(x)[0] > threshold` (sound)
    /// - If `PotentialViolation(ce)`: `ce` is a point where output might be <= threshold
    /// - If `Unknown`: verification exhausted resources (timeout/domain limit)
    /// - Sound: Verified implies no counterexample exists (no false positives)
    #[instrument(skip(self, network, input), fields(threshold, input_shape = ?input.shape(), max_domains = self.config.max_domains))]
    pub fn verify(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
    ) -> Result<BetaCrownResult> {
        self.verify_impl(network, input, threshold, self.engine(), None)
    }

    /// Internal verify implementation with optional GemmEngine for GPU acceleration.
    fn verify_impl(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        self.config.validate()?;
        // Disable the L2/Cauchy–Schwarz lever for the entire beta-CROWN verify
        // scope. beta-CROWN re-runs IBP forward bound collection per domain (root
        // bounds, child clips, input-split prescreen — some on rayon workers via
        // RayonTaskGuard, which also disables the lever there). This is the
        // outer chokepoint for both `verify` and `verify_with_engine`. Sound (the
        // lever only tightens); restored on drop. See `crate::l2_lever_gate`.
        let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
        let start_time = Instant::now();
        // Env-gated throughput probe (#38 acasxu prop_2). Prints domains/s at each
        // BaB loop exit so A/B runs can report explored-domain counts + rate
        // without a tracing subscriber. No effect unless NY_ACASXU_PROF is set.
        let prof = std::env::var("NY_ACASXU_PROF").is_ok();
        let prof_report = |tag: &str,
                           explored: usize,
                           verified: usize,
                           depth: usize,
                           clause_pruned: usize| {
            if prof {
                let secs = start_time.elapsed().as_secs_f64().max(1e-9);
                eprintln!(
                        "[NY_ACASXU_PROF] {tag}: explored={explored} verified={verified} max_depth={depth} clause_pruned={clause_pruned} elapsed={secs:.2}s rate={:.0}/s",
                        explored as f64 / secs
                    );
            }
        };
        let effective_total = match deadline {
            Some(dl) => dl.saturating_duration_since(start_time),
            None => self.config.timeout,
        };
        let crown_deadline = Some(start_time.checked_add(effective_total).ok_or_else(|| {
            NyError::InvalidConfig(format!(
                "effective timeout {:?} is too large for the platform monotonic clock",
                effective_total
            ))
        })?);
        let mut cut_gate = CutGateState::new(&self.config);
        let mut state = BabLoopState::new(self.config.enable_cuts);
        // Conflict-clause learning (win-plan arc C, v1): per-run store, gated
        // NY_BAB_CLAUSE_LEARN=1 (default OFF => byte-identical baseline).
        // Disabled outright under the InputSplit heuristic; per-domain
        // input-split evidence fails closed inside the store regardless.
        // Per-run scope => same network, same root box, same threshold, same
        // objective sense for every recorded clause BY CONSTRUCTION.
        state.clause_store = crate::beta_crown::conflict_clauses::ClauseStore::from_env(!matches!(
            self.config.branching_heuristic,
            crate::beta_crown::branching::BranchingHeuristic::InputSplit
        ));
        if state.clause_store.is_enabled() {
            info!("BaB conflict-clause learning enabled (NY_BAB_CLAUSE_LEARN=1)");
        }

        if cut_gate.is_cold_start() {
            state.cut_generation_enabled = false;
            info!("BICCOS cold-start gating active: cuts disabled until thresholds met");
        }

        let mut cut_pool = if self.config.enable_cuts {
            CutPool::from_config(&self.config)
        } else {
            CutPool::new(0)
        };

        let (initial_bounds, initial_layer_bounds, base_layer_bounds, bab_timeout, pgd_deadline) =
            match self.evaluate_initial_phase(
                network,
                input,
                threshold,
                engine,
                start_time,
                &cut_gate,
                &mut cut_pool,
                crown_deadline,
            )? {
                InitialPhaseOutcome::Early(result) => return Ok(result),
                InitialPhaseOutcome::Proceed {
                    initial_bounds,
                    layer_bounds,
                    base_layer_bounds,
                    bab_timeout,
                    pgd_deadline,
                } => (
                    initial_bounds,
                    layer_bounds,
                    base_layer_bounds,
                    bab_timeout,
                    pgd_deadline,
                ),
            };
        let pgd_fallback = |result| {
            self.try_pgd_attack_with_deadline(network, input, threshold, result, pgd_deadline)
        };

        // BaB queue MEMORY cap (#bab-queue-oom). Each `BabDomain` on the heap carries
        // per-layer intermediate bounds for the whole network; on conv-heavy nets
        // (cifar100 / tinyimagenet ResNets) that is megabytes per domain, so the
        // count-based `max_domains` (50k) lets the heap grow to tens of GB and OOM —
        // and stalls the deadline long before, the reason these benchmarks time out to
        // a 0 score. Bound the QUEUE by BYTES instead: derive a per-domain footprint
        // from the network's total intermediate-bound size and stop expanding once the
        // live queue would exceed the budget, returning a sound `Unknown` (BaB halting
        // early on a memory bound only loosens the result — never makes it unsound).
        // Small models (acasxu/sat_relu, tiny per-domain footprint) get an effective
        // cap far above `max_domains`, so they are unaffected.
        let per_domain_neurons: usize = initial_layer_bounds
            .iter()
            .map(BoundedTensor::len)
            .sum::<usize>()
            .max(1);
        // ~16 B/neuron: lower+upper f32 (8 B) plus alpha/beta/split-history overhead.
        let per_domain_bytes = per_domain_neurons.saturating_mul(16).max(1);
        let queue_mem_budget_bytes = std::env::var("NY_BAB_QUEUE_MEM_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(3072)
            .saturating_mul(1024 * 1024);
        let max_queue_domains = (queue_mem_budget_bytes / per_domain_bytes).max(1);

        let mut queue: BinaryHeap<BabDomain> = BinaryHeap::new();
        let mut root = self.build_root_domain(initial_layer_bounds, &initial_bounds, input)?;
        root.priority = self
            .config
            .violation_priority(root.lower_bound, root.upper_bound)?;
        queue.push(root);

        let batch_size = self.config.batch_size.max(1);
        let mut prof_last = start_time;
        let mut prof_iters: u64 = 0;
        while !queue.is_empty() {
            if prof {
                prof_iters += 1;
                let since = prof_last.elapsed().as_secs_f64();
                if since >= 2.0 {
                    eprintln!(
                        "[NY_ACASXU_PROF] tick: iters={prof_iters} explored={} verified={} queue={} depth={} elapsed={:.1}s",
                        state.domains_explored,
                        state.domains_verified,
                        queue.len(),
                        state.max_depth,
                        start_time.elapsed().as_secs_f64()
                    );
                    prof_last = Instant::now();
                }
            }
            if start_time.elapsed() > bab_timeout {
                info!(
                    "β-CROWN timeout after {} domains, {} verified, {} cuts",
                    state.domains_explored, state.domains_verified, cut_pool.total_generated
                );
                // #bab-frontier: the surviving queue is exactly where a CE must
                // live; export the top-K subboxes as attack seeds before the
                // queue is dropped (env-gated, post-deadline work, guidance
                // only — see bab_frontier_export).
                crate::beta_crown::bab_frontier_export::record_bab_frontier_if_enabled(
                    &queue,
                    input,
                    self.joint_margin_closer(),
                );
                prof_report(
                    "timeout",
                    state.domains_explored,
                    state.domains_verified,
                    state.max_depth,
                    state.domains_clause_pruned,
                );
                return pgd_fallback(BetaCrownResult {
                    result: BabVerificationStatus::Timeout,
                    domains_explored: state.domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: state.max_depth,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified: state.domains_verified,
                });
            }

            if state.domains_explored >= self.config.max_domains {
                info!(
                    "β-CROWN hit domain limit: {}, {} verified, {} cuts",
                    self.config.max_domains, state.domains_verified, cut_pool.total_generated
                );
                // #bab-frontier: export the surviving frontier (see timeout exit).
                crate::beta_crown::bab_frontier_export::record_bab_frontier_if_enabled(
                    &queue,
                    input,
                    self.joint_margin_closer(),
                );
                prof_report(
                    "domain_limit",
                    state.domains_explored,
                    state.domains_verified,
                    state.max_depth,
                    state.domains_clause_pruned,
                );
                return pgd_fallback(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: format!("Domain limit {} reached", self.config.max_domains),
                    },
                    domains_explored: state.domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: state.max_depth,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified: state.domains_verified,
                });
            }

            if queue.len() >= max_queue_domains {
                info!(
                    "β-CROWN BaB queue memory cap: {} live domains × ~{} B/domain reached the {} MB budget (NY_BAB_QUEUE_MEM_MB); returning sound Unknown (prevents OOM on large/conv nets)",
                    queue.len(),
                    per_domain_bytes,
                    queue_mem_budget_bytes / (1024 * 1024)
                );
                // #bab-frontier: export the surviving frontier (see timeout exit).
                crate::beta_crown::bab_frontier_export::record_bab_frontier_if_enabled(
                    &queue,
                    input,
                    self.joint_margin_closer(),
                );
                return pgd_fallback(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: format!(
                            "BaB queue memory budget reached ({} live domains)",
                            queue.len()
                        ),
                    },
                    domains_explored: state.domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: state.max_depth,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified: state.domains_verified,
                });
            }

            // Clamp the batch to the REMAINING domain budget.
            //
            // `prefilter_domain_batch` counts every domain in the batch, but the
            // `max_domains` check above only runs BETWEEN batches, so a full
            // batch could carry `domains_explored` past the cap by up to
            // `batch_size - 1`. With `--max-domains 2` and a 2-child split that
            // reported 3 domains explored against a cap of 2.
            //
            // Overshooting explores MORE than asked, so it was never a soundness
            // problem — it is a resource-discipline one, and `--max-domains` is
            // the knob operators use to bound work. A cap that silently admits
            // `cap + batch_size - 1` is not a cap.
            let remaining_domains = self
                .config
                .max_domains
                .saturating_sub(state.domains_explored);
            let batch = pop_domain_batch(&mut queue, batch_size.min(remaining_domains).max(1));
            if batch.is_empty() {
                break;
            }

            let cuts_active_for_batch = !cut_pool.is_empty() && self.config.enable_cuts;
            let prefilter = self.prefilter_domain_batch(
                batch,
                threshold,
                &mut state,
                &mut cut_pool,
                network,
                input,
                &base_layer_bounds,
                engine,
                start_time,
            )?;
            if let Some(violation) = prefilter.violation {
                return Ok(violation);
            }

            if prefilter.domains_to_process.is_empty() {
                record_cut_gate_batch(
                    &self.config,
                    &mut cut_gate,
                    &mut state,
                    &cut_pool,
                    prefilter.batch_domain_count,
                    prefilter.verified_in_batch,
                    None,
                    cuts_active_for_batch,
                );
                continue;
            }

            let branching = self.process_branching_batch(
                network,
                input,
                &prefilter.domains_to_process,
                threshold,
                batch_size,
                &mut cut_pool,
                engine,
                crown_deadline,
            );
            if branching.had_propagation_failure {
                state.unresolved_due_to_propagation_failure = true;
            }
            if branching.had_no_branch {
                state.unresolved_due_to_no_branch = true;
            }
            if branching.had_unsplittable {
                state.unresolved_due_to_unsplittable = true;
            }

            let verified_children_in_batch = self.process_batch_children(
                branching.child_results,
                threshold,
                &mut queue,
                &mut state,
                &mut cut_pool,
                network,
                input,
                &base_layer_bounds,
                engine,
            )?;

            let bound_gain_avg = if branching.bound_gain_count > 0 {
                Some(branching.bound_gain_sum / branching.bound_gain_count as f32)
            } else {
                None
            };
            let batch_verified_total = prefilter.verified_in_batch + verified_children_in_batch;
            record_cut_gate_batch(
                &self.config,
                &mut cut_gate,
                &mut state,
                &cut_pool,
                prefilter.batch_domain_count + verified_children_in_batch,
                batch_verified_total,
                bound_gain_avg,
                cuts_active_for_batch,
            );
        }

        if state.has_unresolved() {
            let reason = state.unresolved_reason(self.config.max_depth);
            info!(
                "β-CROWN returning Unknown due to unresolved domains: {} (explored={}, verified={}, cuts={})",
                reason,
                state.domains_explored, state.domains_verified, cut_pool.total_generated
            );
            prof_report(
                "unresolved",
                state.domains_explored,
                state.domains_verified,
                state.max_depth,
                state.domains_clause_pruned,
            );
            return Ok(state.unknown_result(start_time, cut_pool.total_generated, reason));
        }

        prof_report(
            "verified",
            state.domains_explored,
            state.domains_verified,
            state.max_depth,
            state.domains_clause_pruned,
        );
        info!(
            "β-CROWN verified after {} domains, {} verified ({} clause-pruned), {} cuts, max depth {}",
            state.domains_explored,
            state.domains_verified,
            state.domains_clause_pruned,
            cut_pool.total_generated,
            state.max_depth
        );
        Ok(state.verified_result(start_time, cut_pool.total_generated))
    }

    pub(in crate::beta_crown::engine) fn bound_gain(
        &self,
        parent: &BabDomain,
        child: &BabDomain,
    ) -> f32 {
        self.config.bound_gain(
            parent.lower_bound,
            parent.upper_bound,
            child.lower_bound,
            child.upper_bound,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::config::BetaCrownConfig;
    use crate::InputSplitBatchRecord;
    use ndarray::arr1;
    use ny_tensor::BoundedTensor;
    use std::time::Duration;

    // ── Constructor / engine tests ─────────────────────────────────

    #[test]
    fn test_new_sets_deadline_from_timeout() {
        let before = Instant::now();
        let config = BetaCrownConfig {
            timeout: Duration::from_secs(30),
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let after = Instant::now();

        let deadline = verifier
            .config
            .alpha_config
            .deadline
            .expect("deadline must be set");
        assert!(deadline >= before + Duration::from_secs(30));
        assert!(deadline <= after + Duration::from_secs(30));
    }

    #[test]
    fn test_new_zero_timeout_sets_immediate_deadline() {
        let before = Instant::now();
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            timeout: Duration::ZERO,
            ..Default::default()
        });
        let after = Instant::now();
        let deadline = verifier
            .config
            .alpha_config
            .deadline
            .expect("zero is a concrete, immediately expired engine budget");
        assert!(deadline >= before);
        assert!(deadline <= after);
        assert!(verifier.past_effective_graph_bab_deadline());
    }

    #[test]
    fn test_new_unrepresentable_timeout_fails_closed_at_construction() {
        let before = Instant::now();
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            timeout: Duration::from_secs(u64::MAX),
            ..Default::default()
        });
        let after = Instant::now();
        let deadline = verifier
            .config
            .alpha_config
            .deadline
            .expect("constructor must retain a fail-closed deadline");
        assert!(deadline >= before);
        assert!(deadline <= after);
        assert!(verifier.past_effective_graph_bab_deadline());
    }

    #[test]
    fn test_verify_rejects_unrepresentable_timeout_without_panicking() {
        let w = ndarray::arr2(&[[1.0]]);
        let linear = crate::LinearLayer::new(w, None).expect("valid linear");
        let mut network = Network::new();
        network.add_layer(crate::Layer::Linear(linear));
        let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid input");
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            timeout: Duration::from_secs(u64::MAX),
            ..Default::default()
        });

        match verifier.verify(&network, &input, 0.0) {
            Err(NyError::InvalidConfig(message)) => {
                assert!(message.contains("too large for the platform monotonic clock"));
            }
            Err(other) => panic!("expected InvalidConfig, got {other:?}"),
            Ok(_) => panic!("an unrepresentable timeout must fail before verification"),
        }
    }

    #[test]
    fn effective_graph_bab_deadline_uses_earliest_active_scope_and_restores() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            timeout: Duration::from_secs(30),
            ..Default::default()
        });
        let configured = verifier
            .config
            .alpha_config
            .deadline
            .expect("constructor deadline");
        let reserved_bab_deadline = Instant::now() + Duration::from_secs(5);

        assert_eq!(verifier.effective_graph_bab_deadline(), Some(configured));
        {
            let _scope = verifier
                .complete_clip_deadline_overrides
                .scoped(Some(reserved_bab_deadline));
            assert_eq!(
                verifier.effective_graph_bab_deadline(),
                Some(reserved_bab_deadline)
            );
        }
        assert_eq!(verifier.effective_graph_bab_deadline(), Some(configured));

        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("expired test deadline");
        {
            let _scope = verifier
                .complete_clip_deadline_overrides
                .scoped(Some(expired));
            assert!(verifier.past_effective_graph_bab_deadline());
        }
        assert!(!verifier.past_effective_graph_bab_deadline());
    }

    #[test]
    fn test_new_with_engine_stores_engine() {
        let config = BetaCrownConfig::default();
        let engine = Arc::new(ny_core::NaiveCpuGemmEngine);
        let verifier = BetaCrownVerifier::new_with_engine(config, engine);
        assert!(verifier.engine().is_some());
        assert!(verifier.engine_arc().is_some());
    }

    #[test]
    fn test_new_without_engine() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        assert!(verifier.engine().is_none());
        assert!(verifier.engine_arc().is_none());
    }

    struct TestMetricsSink;

    impl InputSplitMetricsSink for TestMetricsSink {
        fn record_batch_summary(&self, _record: &InputSplitBatchRecord) -> Result<()> {
            Ok(())
        }
    }

    impl GraphDomainBatchMetricsSink for TestMetricsSink {
        fn record_batch_summary(
            &self,
            _record: &super::super::graph::domain_batch::GraphDomainBatchRecord,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_with_config_from_inherits_engine() {
        let engine = Arc::new(ny_core::NaiveCpuGemmEngine);
        let original = BetaCrownVerifier::new_with_engine(BetaCrownConfig::default(), engine);

        let new_config = BetaCrownConfig {
            timeout: Duration::from_secs(99),
            ..Default::default()
        };
        let derived = original.with_config_from(new_config);

        assert!(derived.engine().is_some(), "engine must be inherited");
        assert_eq!(derived.config.timeout, Duration::from_secs(99));
    }

    #[test]
    fn test_with_input_split_metrics_sink_stores_sink() {
        let sink: Arc<dyn InputSplitMetricsSink> = Arc::new(TestMetricsSink);
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default())
            .with_input_split_metrics_sink(sink.clone());

        assert!(verifier.input_split_metrics_sink().is_some());
        assert!(verifier.input_split_metrics_sink_arc().is_some());
        assert!(Arc::ptr_eq(
            &verifier
                .input_split_metrics_sink_arc()
                .expect("stored metrics sink"),
            &sink,
        ));
    }

    #[test]
    fn test_with_config_from_inherits_input_split_metrics_sink() {
        let sink: Arc<dyn InputSplitMetricsSink> = Arc::new(TestMetricsSink);
        let original =
            BetaCrownVerifier::new(BetaCrownConfig::default()).with_input_split_metrics_sink(sink);

        let derived = original.with_config_from(BetaCrownConfig {
            timeout: Duration::from_secs(7),
            ..Default::default()
        });

        assert!(
            derived.input_split_metrics_sink().is_some(),
            "metrics sink must be inherited"
        );
        assert!(Arc::ptr_eq(
            &derived
                .input_split_metrics_sink_arc()
                .expect("derived metrics sink"),
            &original
                .input_split_metrics_sink_arc()
                .expect("original metrics sink"),
        ));
    }

    #[test]
    fn test_with_graph_domain_batch_metrics_sink_stores_sink() {
        let sink: Arc<dyn GraphDomainBatchMetricsSink> = Arc::new(TestMetricsSink);
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default())
            .with_graph_domain_batch_metrics_sink(sink.clone());

        assert!(verifier.graph_domain_batch_metrics_sink().is_some());
        assert!(verifier.graph_domain_batch_metrics_sink_arc().is_some());
        assert!(Arc::ptr_eq(
            &verifier
                .graph_domain_batch_metrics_sink_arc()
                .expect("stored graph-domain metrics sink"),
            &sink,
        ));
    }

    #[test]
    fn test_with_config_from_inherits_graph_domain_batch_metrics_sink() {
        let sink: Arc<dyn GraphDomainBatchMetricsSink> = Arc::new(TestMetricsSink);
        let original = BetaCrownVerifier::new(BetaCrownConfig::default())
            .with_graph_domain_batch_metrics_sink(sink);

        let derived = original.with_config_from(BetaCrownConfig {
            timeout: Duration::from_secs(11),
            ..Default::default()
        });

        assert!(
            derived.graph_domain_batch_metrics_sink().is_some(),
            "graph-domain metrics sink must be inherited"
        );
        assert!(Arc::ptr_eq(
            &derived
                .graph_domain_batch_metrics_sink_arc()
                .expect("derived graph-domain metrics sink"),
            &original
                .graph_domain_batch_metrics_sink_arc()
                .expect("original graph-domain metrics sink"),
        ));
    }

    #[test]
    fn test_with_config_from_no_engine() {
        let original = BetaCrownVerifier::new(BetaCrownConfig::default());
        let derived = original.with_config_from(BetaCrownConfig {
            timeout: Duration::from_secs(42),
            ..Default::default()
        });
        assert!(derived.engine().is_none());
        assert_eq!(derived.config.timeout, Duration::from_secs(42));
    }

    #[test]
    fn test_resolve_engine_prefers_argument() {
        let stored = Arc::new(ny_core::NaiveCpuGemmEngine);
        let verifier = BetaCrownVerifier::new_with_engine(BetaCrownConfig::default(), stored);

        let arg_engine = ny_core::NaiveCpuGemmEngine;
        let resolved = verifier.resolve_engine(Some(&arg_engine));
        assert!(resolved.is_some());
    }

    #[test]
    fn test_resolve_engine_falls_back_to_stored() {
        let stored = Arc::new(ny_core::NaiveCpuGemmEngine);
        let verifier = BetaCrownVerifier::new_with_engine(BetaCrownConfig::default(), stored);
        let resolved = verifier.resolve_engine(None);
        assert!(resolved.is_some());
    }

    #[test]
    fn test_resolve_engine_none_when_neither() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let resolved = verifier.resolve_engine(None);
        assert!(resolved.is_none());
    }

    // ── verify early-exit tests ────────────────────────────────────

    #[test]
    fn test_verify_root_verified_returns_immediately() {
        let w = ndarray::arr2(&[[1.0]]);
        let linear = crate::LinearLayer::new(w, None).expect("valid linear");
        let mut network = Network::new();
        network.add_layer(crate::Layer::Linear(linear));

        let input = BoundedTensor::new(arr1(&[10.0_f32]).into_dyn(), arr1(&[20.0_f32]).into_dyn())
            .expect("valid input");

        // Identity network: output = input. Lower bound is 10.0.
        // Threshold = 5.0 → root should verify immediately (10 > 5).
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let result = verifier
            .verify(&network, &input, 5.0)
            .expect("verify should succeed");
        assert_eq!(result.result, BabVerificationStatus::Verified);
        assert_eq!(result.domains_explored, 1);
        assert_eq!(result.domains_verified, 1);
    }

    /// `input_split_depth = 0` selects no split dimensions, so input-split
    /// BaB could never branch. The sequential entry must reject the config
    /// up front (same as the graph lanes) instead of silently running a
    /// search that can only drop domains — for out = x on [-5, 5] the
    /// property `out > 0` is false on the whole half x < 0, so anything
    /// short of a hard error risks masking that.
    #[test]
    fn test_verify_input_split_depth_zero_rejected() {
        use crate::beta_crown::branching::BranchingHeuristic;
        use ny_core::NyError;
        let w = ndarray::arr2(&[[1.0]]);
        let linear = crate::LinearLayer::new(w, None).expect("valid linear");
        let mut network = Network::new();
        network.add_layer(crate::Layer::Linear(linear));

        let input = BoundedTensor::new(arr1(&[-5.0_f32]).into_dyn(), arr1(&[5.0_f32]).into_dyn())
            .expect("valid input");

        let config = BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::InputSplit,
            input_split_depth: 0,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let err = verifier
            .verify(&network, &input, 0.0)
            .expect_err("depth-0 InputSplit config must be rejected at the verify entry");
        assert!(
            matches!(err, NyError::InvalidConfig(_)),
            "expected NyError::InvalidConfig, got {err:?}"
        );
        assert!(
            err.to_string().contains("input_split_depth"),
            "error should name the offending field: got '{err}'"
        );
    }

    /// A fully-degenerate (point) input box has no positive-width dimension
    /// to split. With out = x and threshold 0, the root bounds are exactly
    /// [0, 0]: not strictly above the threshold, so the property is unproven
    /// and dropping the unsplittable root must yield Unknown, not Verified.
    #[test]
    fn test_verify_input_split_point_box_returns_unknown() {
        use crate::beta_crown::branching::BranchingHeuristic;
        let w = ndarray::arr2(&[[1.0]]);
        let linear = crate::LinearLayer::new(w, None).expect("valid linear");
        let mut network = Network::new();
        network.add_layer(crate::Layer::Linear(linear));

        let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[0.0_f32]).into_dyn())
            .expect("valid input");

        let config = BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::InputSplit,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let result = verifier.verify(&network, &input, 0.0).expect("verify");
        assert!(
            !matches!(result.result, BabVerificationStatus::Verified),
            "point box with output == threshold is unproven; got {:?}",
            result.result
        );
        assert!(
            matches!(result.result, BabVerificationStatus::Unknown { .. }),
            "unsplittable point box must yield Unknown, got {:?}",
            result.result
        );
    }

    #[test]
    fn test_verify_root_violation_returns_immediately() {
        let w = ndarray::arr2(&[[1.0]]);
        let linear = crate::LinearLayer::new(w, None).expect("valid linear");
        let mut network = Network::new();
        network.add_layer(crate::Layer::Linear(linear));

        let input = BoundedTensor::new(arr1(&[-5.0_f32]).into_dyn(), arr1(&[-1.0_f32]).into_dyn())
            .expect("valid input");

        // Identity network: output = input. Upper bound is -1.0.
        // Threshold = 0.0 → root should detect violation (-1 < 0).
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let result = verifier
            .verify(&network, &input, 0.0)
            .expect("verify should succeed");
        assert_eq!(result.result, BabVerificationStatus::potential_violation());
        assert_eq!(result.domains_explored, 1);
        assert_eq!(result.domains_verified, 0);
    }
}
