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
use crate::beta_crown::result::{
    note_global_timeout_engine_invocation, record_global_timeout_deadline_source,
    BabVerificationStatus, BetaCrownResult,
};
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
        // #vit-collector-throughput / #vit-batched-bilinear: deliver the typed
        // collector-kernel keys (`bab.bilinear_rowpar_kernel`,
        // `bab.linear_position_stack`, `bab.reciprocal_unbounded_upper`,
        // `bab.bilinear_gemm_kernel`) to the layer kernels, which sit under
        // the configuration-free backward dispatcher. STICKY-ON: an unset key
        // never disarms, and an all-default config (every shipped preset)
        // touches nothing — see `layers::collector_kernels`.
        crate::layers::collector_kernels::arm(
            crate::layers::collector_kernels::CollectorKernelArming {
                bilinear_rowpar: config.bilinear_rowpar_kernel,
                linear_pos_stack: config.linear_position_stack,
                reciprocal_unbounded: config.reciprocal_unbounded_upper,
                bilinear_gemm: config.bilinear_gemm_kernel,
            },
        );
        // `BetaCrownConfig::timeout` is the verifier's concrete engine budget:
        // zero therefore means an immediately expired verifier. The CLI maps
        // its user-facing zero/unbounded sentinel to a representable long
        // engine horizon before construction. Use checked arithmetic so a
        // direct caller's platform-unrepresentable duration cannot panic.
        let now = Instant::now();
        // CONSEQUENCE WORTH KNOWING (historical; the UPDATE below records the fix):
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
        //
        // UPDATE: both halves are now fixed. #coop-chain-alpha-grad runs the
        // replay under a private slice (`chain_gradient_budget`; the log line is
        // `[chain-alpha-grad] armed: slice_ms=...`), and #alpha-envelope-grad-cpu
        // threads the TYPED `alpha_config.alpha_envelope_grad` into the replay's
        // CPU chain gradient, which previously resolved env-only, so the preset
        // key `bab.alpha_crown.alpha_envelope_grad` is now THREADED to this lane.
        // That is delivery, not arrival: arrival on a scored row is unmeasured
        // until that row prints the unconditional `[alpha-envelope-grad] cpu=1
        // ... source=config` line with no NY_* set.
        config.alpha_config.deadline = Some(now.checked_add(config.timeout).unwrap_or(now));
        Self {
            config,
            engine,
            input_split_metrics_sink,
            graph_domain_batch_metrics_sink,
            joint_margin_closer: None,
            input_head_continuation: None,
            graph_mip_leaf_oracle: None,
            // #vit-margin-alpha-complex (P8): attached only by an armed CLI.
            root_adjudication_oracle: None,
            // #relusplitter-lp-refine: attached only by an armed CLI.
            root_interm_refiner: None,
            // #split-hint: empty until a root publishes its influence vector.
            // Disarmed (`SPLIT_HINT == false`) nothing ever writes it.
            root_split_hint: Default::default(),
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
            graph_cover: Default::default(),
            la_warm_frontier_counters: Default::default(),
            kfsb_single_shot_telemetry: Default::default(),
            mo_cover: Default::default(),
        }
    }

    /// Create a new β-CROWN verifier.
    ///
    /// Sets `alpha_config.deadline` from `timeout` so that per-domain beta
    /// optimization loops can bail early when the verification budget is
    /// exhausted. Without this, `past_deadline()` always returns `false`
    /// and a single BaB batch can exceed the total budget by 2-10x.
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
    /// acceleration without per-call engine threading.
    pub fn with_config_from(&self, config: BetaCrownConfig) -> Self {
        let mut v = Self::new_with_resources(
            config,
            self.engine_arc(),
            self.input_split_metrics_sink_arc(),
            self.graph_domain_batch_metrics_sink_arc(),
        );
        v.joint_margin_closer = self.joint_margin_closer.clone();
        v.input_head_continuation = self.input_head_continuation.clone();
        v.graph_mip_leaf_oracle = self.graph_mip_leaf_oracle.clone();
        v.root_adjudication_oracle = self.root_adjudication_oracle.clone();
        v.root_interm_refiner = self.root_interm_refiner.clone();
        v.disjunctive_restart_root_cache = self.disjunctive_restart_root_cache.clone();
        // #la-warm-frontier: one lane, one set of counters.
        v.la_warm_frontier_counters = Arc::clone(&self.la_warm_frontier_counters);
        // #kfsb-production: one lane, one line budget, one fan-out lease.
        v.kfsb_single_shot_telemetry = Arc::clone(&self.kfsb_single_shot_telemetry);
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

    /// Attach the independent-forward oracle (#vit-margin-alpha-complex, P8)
    /// the default-off containment fence uses to adjudicate a ROOT-terminal
    /// multi-objective `Verified` (see
    /// [`RootAdjudicationOracle`](crate::beta_crown::root_adjudication::RootAdjudicationOracle)).
    /// Consulted only when BOTH `mo_containment_fence` and
    /// `mo_root_ort_adjudication` are armed; it can only downgrade a verdict.
    /// Runtime-only; `None` (never attached) is byte-for-byte inert. Preserved
    /// across `with_config_from`.
    #[must_use]
    pub fn with_root_adjudication_oracle(
        mut self,
        oracle: Arc<dyn crate::beta_crown::root_adjudication::RootAdjudicationOracle>,
    ) -> Self {
        self.root_adjudication_oracle = Some(oracle);
        self
    }

    /// The attached root-adjudication oracle, if any (see
    /// [`with_root_adjudication_oracle`](Self::with_root_adjudication_oracle)).
    pub(crate) fn root_adjudication_oracle(
        &self,
    ) -> Option<&dyn crate::beta_crown::root_adjudication::RootAdjudicationOracle> {
        self.root_adjudication_oracle.as_deref()
    }

    /// Attach the ROOT intermediate-bound refiner (#relusplitter-lp-refine,
    /// wave-6 C7; see
    /// [`RootIntermRefiner`](crate::beta_crown::root_interm_refine::RootIntermRefiner)).
    /// The multi-objective graph root consults it once, after the bootstrap,
    /// and intersects its certified candidates shrink-only into the stored
    /// node boxes. Runtime-only; `None` (never attached) is byte-for-byte
    /// inert. Preserved across `with_config_from`.
    #[must_use]
    pub fn with_root_interm_refiner(
        mut self,
        refiner: Arc<dyn crate::beta_crown::root_interm_refine::RootIntermRefiner>,
    ) -> Self {
        self.root_interm_refiner = Some(refiner);
        self
    }

    /// The attached root intermediate-bound refiner, if any (see
    /// [`with_root_interm_refiner`](Self::with_root_interm_refiner)).
    pub(crate) fn root_interm_refiner(
        &self,
    ) -> Option<&dyn crate::beta_crown::root_interm_refine::RootIntermRefiner> {
        self.root_interm_refiner.as_deref()
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

    /// Attach the concrete original singleton continuation. Numerical source
    /// binding is checked by the consuming input-split route; generic leaves
    /// do not become original-instance evidence merely through this API.
    #[must_use]
    pub fn with_input_head_continuation(
        mut self,
        continuation: Arc<crate::margin_row::InputHeadContinuation>,
    ) -> Self {
        self.input_head_continuation = Some(continuation);
        self
    }

    pub(crate) fn input_head_continuation(
        &self,
    ) -> Option<&crate::margin_row::InputHeadContinuation> {
        self.input_head_continuation.as_deref()
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
        configured.set_chunk_wave_parallel_enabled(self.config.chunk_wave_parallel_enabled);
        configured.set_frozen_levers(self.config.frozen_levers.clone());
        configured.set_forward_linear_deadline_fallback_to_ibp(
            self.config
                .alpha_config
                .forward_linear_deadline_fallback_to_ibp,
        );
        // #vit-zono-root (DARK, typed `bab.vit_zono_root`): STICKY-ON stamp of
        // the certified forward-zonotope root-map plan. An unarmed config never
        // clears a plan the CLI stamped (a sub-verifier built from a partial
        // config cannot turn the lane dark mid-run), and an unarmed config on
        // an unstamped graph (every shipped preset) touches nothing.
        if let Some(plan) = crate::zono_core::vit_zono_root_plan(&self.config) {
            configured.set_vit_zono_root_plan(Some(plan));
        }
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
    /// remaining wall-clock time instead of `self.config.timeout`.
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
        // #phase-cost: see `ny_core::phase_cost` (observability only).
        let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::BabLoop);
        let mut result = self.verify_impl_inner(network, input, threshold, engine, deadline)?;
        if matches!(result.result, BabVerificationStatus::Verified) {
            if let Some(source) = self.input_head_continuation() {
                if source.dependent_use() {
                    let lo = input.lower().iter().copied().collect::<Vec<_>>();
                    let hi = input.upper().iter().copied().collect::<Vec<_>>();
                    let bound_deadline = deadline
                        .or(self.config.alpha_config.deadline)
                        .unwrap_or_else(Instant::now);
                    if threshold != 0.0
                        || self.config.verify_upper_bound
                        || !source.matches_scalar_network(network)
                        || !source.root_box_matches(&lo, &hi)
                        || !source.has_complete_source_cover(bound_deadline)
                    {
                        result.result = BabVerificationStatus::Unknown {
                            reason: "source-dependent singleton search lacks a complete original input-box cover".into(),
                        };
                    }
                }
            }
        }
        Ok(result)
    }

    fn verify_impl_inner(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        network.ensure_proof_eligible()?;
        self.config.validate()?;
        note_global_timeout_engine_invocation();
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
        let prof = ny_levers::read_raw(&ny_levers::decls::bab_diagnostics::ACASXU_PROF).is_some();
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
        let crown_deadline = match deadline {
            Some(deadline) => Some(deadline),
            None => Some(start_time.checked_add(self.config.timeout).ok_or_else(|| {
                NyError::InvalidConfig(format!(
                    "effective timeout {:?} is too large for the platform monotonic clock",
                    self.config.timeout
                ))
            })?),
        };
        // #lowdim-prover (typed opt-in `bab.branching.input_split.lowdim_prover`):
        // this lane proves `network(x)[0] > threshold` (or `< threshold` in
        // upper-bound mode — the lane proves that as `lower(-y0) > -threshold`,
        // an exact negation). On a small-input network, try the sound
        // forward-linear lane first on a Graph copy of the network with half of
        // the budget. Only a proof returns early; Unproved/Declined continue
        // into this lane unchanged. A single-constraint spec arrives here in
        // upper-bound mode (verify_standard), which an earlier `!upper` gate
        // skipped SILENTLY — measured as a VOID tll probe.
        let lowdim_input_split = matches!(
            self.config.branching_heuristic,
            crate::beta_crown::branching::BranchingHeuristic::InputSplit
        );
        if self.config.input_split_lowdim_prover {
            // The lever's own observable: armed, which entry, and every gate.
            eprintln!(
                "[lowdim] armed=1 entry=sequential input_split={} upper={}",
                u8::from(lowdim_input_split),
                u8::from(self.config.verify_upper_bound)
            );
        }
        if self.config.input_split_lowdim_prover && lowdim_input_split {
            let mut candidate: Option<Vec<f64>> = None;
            let outcome = match GraphNetwork::from_sequential(network) {
                Ok(graph) => {
                    let end = crown_deadline.unwrap_or(start_time + self.config.timeout);
                    let lane_deadline = start_time
                        + end
                            .saturating_duration_since(start_time)
                            .mul_f32(self.config.lowdim_prover_share());
                    crate::lowdim::prove_first_output_witnessed(
                        &graph,
                        input,
                        threshold,
                        self.config.verify_upper_bound,
                        lane_deadline,
                        &self.config.lowdim_config(),
                        &mut candidate,
                    )
                }
                Err(e) => {
                    crate::lowdim::LowDimOutcome::Declined(format!("from_sequential failed: {e}"))
                }
            };
            crate::lowdim::emit_line(0, &outcome);
            if let crate::lowdim::LowDimOutcome::Verified(stats) = outcome {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Verified,
                    domains_explored: usize::try_from(stats.domains).unwrap_or(usize::MAX),
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: stats.max_depth as usize,
                    output_bounds: None,
                    cuts_generated: 0,
                    domains_verified: usize::try_from(stats.closed).unwrap_or(usize::MAX),
                });
            }
            // A Candidate stop hands back the leaf centre the lane's f64
            // lowering scored as violating. Re-evaluate it on THIS network's
            // own f32 concrete forward; only a point that violates there is
            // carried out, as a PotentialViolation witness, so the post-BaB
            // confirmer re-checks it against the full VNN-LIB property and the
            // trusted ONNX-Runtime gate still decides any scored `sat`. A point
            // that does not re-verify is dropped and the lane below runs
            // exactly as before.
            if let (crate::lowdim::LowDimOutcome::Unproved(stats, _), Some(point)) =
                (&outcome, candidate)
            {
                if let Some(witness) = lowdim_candidate_witness(
                    network,
                    input,
                    &point,
                    threshold,
                    self.config.verify_upper_bound,
                ) {
                    eprintln!(
                        "[lowdim] witness=confirmed output0={:e} threshold={threshold:e}",
                        witness.output[0]
                    );
                    return Ok(BetaCrownResult {
                        result: BabVerificationStatus::potential_violation_with(witness),
                        domains_explored: usize::try_from(stats.domains).unwrap_or(usize::MAX),
                        time_elapsed: start_time.elapsed(),
                        max_depth_reached: stats.max_depth as usize,
                        output_bounds: None,
                        cuts_generated: 0,
                        domains_verified: usize::try_from(stats.closed).unwrap_or(usize::MAX),
                    });
                }
                eprintln!("[lowdim] witness=rejected (f32 concrete forward does not violate)");
            }
        }
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
        let bab_deadline = start_time.checked_add(bab_timeout).ok_or_else(|| {
            NyError::InvalidConfig(
                "BaB timeout is too large for the platform monotonic clock".to_string(),
            )
        })?;

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
        // SEED THE ROOT (#bab-warm-start). `BabDomain::root` carries an EMPTY
        // α state and EMPTY intermediate bounds, so every depth-1 child (a)
        // warm-started from nothing, re-deriving the u>-l heuristic, and (b)
        // fell into the full-recompute fallback in `create_child_domain`
        // (parent intermediates empty). One joint-α optimization at the root
        // (empty history: no β, pure α) populates both: children inherit an
        // OPTIMIZED α trajectory and take the frozen-prefix transfer path
        // from the first split on. Measured motivation: with the monotone
        // floor's counter, 100% of children regressed below their parent's
        // bound (clamp 1942/1942 on `mnist_fc 256x4`) — the cold-started
        // per-child optimizer could not even reproduce the root bound.
        // SOUNDNESS: the joint optimizer's root bound and the initial-phase
        // bound both enclose the same region, so `max` is sound; α seeding
        // affects only the optimizer's starting point.
        if self.config.use_alpha_crown
            && !matches!(
                self.config.branching_heuristic,
                crate::beta_crown::branching::BranchingHeuristic::InputSplit
            )
        {
            let root_history = crate::beta_crown::SplitHistory::new();
            let mut root_alpha =
                crate::beta_crown::state::DomainAlphaState::from_layer_bounds_and_constraints(
                    network,
                    &root.layer_bounds,
                    &root_history,
                );
            if !root_alpha.is_empty() {
                let mut root_beta = crate::beta_crown::state::BetaState::empty();
                let (root_joint_bounds, root_intermediate) = self.optimize_joint_bounds(
                    network,
                    input,
                    &root_history,
                    &root.layer_bounds,
                    &mut root_beta,
                    &mut root_alpha,
                    &mut cut_pool,
                    engine,
                )?;
                let joint_lower = {
                    use super::tensor_ext::BoundedTensorExt as _;
                    root_joint_bounds.lower_scalar()
                };
                if prof {
                    eprintln!(
                        "[NY_ACASXU_PROF] root-seed: joint_lower={joint_lower:.4} initial_lower={:.4}",
                        root.lower_bound
                    );
                }
                if joint_lower.is_finite() {
                    root.lower_bound = root.lower_bound.max(joint_lower);
                }
                root.domain_alpha_state = root_alpha;
                root.intermediate_bounds = root_intermediate;
            }
        }
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
                        "[NY_ACASXU_PROF] tick: iters={prof_iters} explored={} verified={} queue={} depth={} elapsed={:.1}s clamp={}/{}",
                        state.domains_explored,
                        state.domains_verified,
                        queue.len(),
                        state.max_depth,
                        start_time.elapsed().as_secs_f64(),
                        super::domain::child::MONOTONE_CLAMP_FIRED
                            .load(std::sync::atomic::Ordering::Relaxed),
                        super::domain::child::CHILDREN_CREATED
                            .load(std::sync::atomic::Ordering::Relaxed),
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
                record_global_timeout_deadline_source(Some(bab_deadline));
                let result = pgd_fallback(BetaCrownResult {
                    result: BabVerificationStatus::Timeout,
                    domains_explored: state.domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: state.max_depth,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified: state.domains_verified,
                })?;
                return Ok(result);
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

/// Round a #lowdim-prover candidate (f64 input coordinates) into the f32 input
/// box and keep it only if THIS network's own concrete f32 forward violates the
/// sequential contract there (`output[0] <= threshold`, or `>= threshold` in
/// upper-bound mode). Rounding then clamping keeps the point inside the box.
/// `None` means the candidate did not survive; it never means the property holds.
fn lowdim_candidate_witness(
    network: &Network,
    input: &BoundedTensor,
    point: &[f64],
    threshold: f32,
    upper: bool,
) -> Option<crate::beta_crown::result::ViolationWitness> {
    let (lo, hi) = (input.lower(), input.upper());
    if point.len() != lo.len() {
        return None;
    }
    let x: Vec<f32> = point
        .iter()
        .zip(lo.iter().zip(hi.iter()))
        .map(|(&p, (&l, &h))| (p as f32).clamp(l, h))
        .collect();
    if x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(input.shape()), x.clone()).ok()?;
    let out = network
        .propagate_concrete_point(&BoundedTensor::concrete(arr).ok()?, None)
        .ok()?
        .center();
    let y0 = *out.iter().next()?;
    // The same margin the lane's candidate test uses: a boundary point is
    // not carried (ONNX Runtime may evaluate it on the other side).
    let margin = crate::lowdim::candidate_margin(f64::from(threshold));
    let violates = if upper {
        f64::from(y0) >= f64::from(threshold) + margin
    } else {
        f64::from(y0) <= f64::from(threshold) - margin
    };
    violates.then(|| crate::beta_crown::result::ViolationWitness {
        input: x,
        input_shape: input.shape().to_vec(),
        output: out.iter().copied().collect(),
    })
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

    fn identity_1d() -> (Network, BoundedTensor) {
        let linear = crate::LinearLayer::new(ndarray::arr2(&[[1.0_f32]]), None).expect("linear");
        let mut network = Network::new();
        network.add_layer(crate::Layer::Linear(linear));
        let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid input");
        (network, input)
    }

    /// The #lowdim-prover candidate is carried out only when the network's own
    /// f32 forward violates at the (box-clamped) point, in the right direction
    /// for the mode.
    #[test]
    fn lowdim_candidate_witness_is_kept_only_when_the_real_forward_violates() {
        let (net, input) = identity_1d();
        let w = lowdim_candidate_witness(&net, &input, &[0.25], 0.5, false).expect("0.25 <= 0.5");
        assert_eq!(w.input, vec![0.25]);
        assert!(w.output[0] <= 0.5);
        assert!(lowdim_candidate_witness(&net, &input, &[0.75], 0.5, false).is_none());
        // upper mode: a violation is y0 >= t
        assert!(lowdim_candidate_witness(&net, &input, &[0.75], 0.5, true).is_some());
        assert!(lowdim_candidate_witness(&net, &input, &[0.25], 0.5, true).is_none());
        // out-of-box coordinates are clamped INTO the box before evaluation
        let w = lowdim_candidate_witness(&net, &input, &[-3.0], 0.5, false).expect("clamps to 0");
        assert_eq!(w.input, vec![0.0]);
        assert!(lowdim_candidate_witness(&net, &input, &[2.0], 0.5, false).is_none());
        // a point of the wrong dimension is refused, never padded
        assert!(lowdim_candidate_witness(&net, &input, &[0.1, 0.1], 0.5, false).is_none());
    }

    /// End to end through `verify`: armed, the lane proves a true margin and
    /// hands a violated one to the confirmer as a CARRIED witness, never as a
    /// verdict.
    #[test]
    fn armed_lowdim_lane_proves_a_true_margin_and_carries_a_witness_for_a_false_one() {
        let (net, input) = identity_1d();
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            timeout: Duration::from_secs(30),
            branching_heuristic: crate::beta_crown::branching::BranchingHeuristic::InputSplit,
            input_split_lowdim_prover: true,
            ..Default::default()
        });
        let proved = verifier.verify(&net, &input, -0.5).expect("runs");
        assert_eq!(proved.result, BabVerificationStatus::Verified);
        let found = verifier.verify(&net, &input, 0.5).expect("runs");
        let witness = found
            .result
            .potential_violation_witness()
            .unwrap_or_else(|| panic!("expected a carried witness, got {:?}", found.result));
        assert_eq!(witness.input.len(), 1);
        assert!((0.0..=0.5).contains(&witness.input[0]), "{witness:?}");
        assert!(witness.output[0] <= 0.5, "{witness:?}");
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
