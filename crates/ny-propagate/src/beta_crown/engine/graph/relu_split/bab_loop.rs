// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core BaB loop for ReLU-splitting branch-and-bound.
//!
//! Contains the main `verify_graph_relu_split_impl` method decomposed into
//! focused helper methods for initialization plus the branch-and-bound
//! orchestration. Status checks, aggregation, child evaluation, and domain
//! filtering live in sibling modules.

use std::collections::BinaryHeap;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;
use tracing::{debug, info, instrument};

use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::graph::adaptive_microbatch::{
    adaptive_microbatch_controller_enabled, estimate_graph_domain_bytes, AdaptiveBatchRoute,
    AdaptiveMicrobatchController, MicrobatchMemoryBudget, OrderedBatchCursor, RefusalAction,
};
use crate::beta_crown::engine::graph::domain_batch::{
    GraphDomainBatchEmitTiming, GraphDomainBatchExecutionMode, GraphDomainBatchExecutor,
    GraphDomainBatchPlan, SingleObjectiveBatchRequest,
};
use crate::beta_crown::engine::graph::shared::init::{
    compute_graph_bab_bootstrap, compute_graph_root_output_bounds,
};
use crate::beta_crown::engine::graph::shared::setup::{
    build_graph_bab_setup, build_graph_cut_pool, build_root_alpha_state,
};
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::faer_parallelism::RayonTaskGuard;
use crate::GraphNetwork;

use super::super::super::domain_results::GraphDomainResult;
use super::super::super::BetaCrownVerifier;
use super::super::objectives::objective_bounds;
use super::domain_filter::PreFilterOutcome;

/// Initial bounds for the BaB loop: (node_bounds, optional_alpha_state, root_lower, root_upper).
type InitialBounds = (
    std::collections::HashMap<String, BoundedTensor>,
    Option<crate::bounds::GraphAlphaState>,
    f32,
    f32,
);

fn initial_bounds_deadline_status(
    now: Instant,
    bootstrap_deadline: Option<Instant>,
) -> BabVerificationStatus {
    if bootstrap_deadline.is_some_and(|deadline| now >= deadline) {
        BabVerificationStatus::Timeout
    } else {
        BabVerificationStatus::Unknown {
            reason: "Initial-bound warmup exceeded its deadline cap before branching".to_string(),
        }
    }
}

impl BetaCrownVerifier {
    /// Internal GraphNetwork ReLU split implementation with optional GemmEngine.
    ///
    /// `deadline`: If `Some`, the engine derives its phase budgets from remaining
    /// wall-clock time (`deadline - now`) instead of `self.config.timeout`. This
    /// accounts for time consumed by pre-BaB phases (PGD, IBP). Part of #4321.
    #[instrument(skip(self, graph, input, objective, engine, deadline), fields(threshold, input_shape = ?input.shape(), num_nodes = graph.nodes.len()))]
    pub(crate) fn verify_graph_relu_split_impl(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        let graph = self.configured_graph_for_crown(graph);
        let graph = &graph;
        let now = Instant::now();
        let mut lifecycle = GraphBabLifecycle::new(now);

        // Steps 1-2: Collect initial node bounds and compute initial output bounds
        // Thread deadline so α-CROWN bails early if timeout budget exhausted (#2698)
        //
        // Warmup cap (#2206 Packet C, #4095): initial bounds get at most
        // `initial_bounds_fraction` of the BaB timeout. Without this cap,
        // alpha-CROWN warmup consumes the entire budget on graph models
        // (soundnessbench, cifar100, tinyimagenet) and the BaB loop never starts.
        //
        // When a wall-clock deadline is provided (#4321), derive the effective
        // timeout from remaining time instead of the configured timeout.
        let pgd_frac = self
            .config
            .phase_budget
            .post_bab_pgd_fraction
            .clamp(0.0, 0.5);
        let effective_total = match deadline {
            Some(dl) => dl.saturating_duration_since(now),
            None => self.config.timeout,
        };
        let bab_timeout = effective_total.mul_f32(1.0 - pgd_frac);
        // The mandatory foundational node-bounds sweep (IBP/CROWN-IBP) must visit
        // every node before BaB can start, so it gets the full global deadline —
        // capping it at `initial_bounds_fraction` choked conv-heavy DAGs (yolo,
        // tinyimagenet) into "deadline exceeded before node 'Conv_0'" with most of
        // the budget unused (#4321). The warmup cap is retained only for the
        // genuinely-iterative root α-CROWN output-bound optimization below.
        let bootstrap_deadline = Some(now + bab_timeout);
        let iterative_deadline = {
            let frac = self
                .config
                .phase_budget
                .initial_bounds_fraction
                .clamp(0.0, 1.0);
            Some(now + bab_timeout.mul_f32(frac))
        };
        let (initial_node_bounds, root_alpha_state, root_lower, root_upper) = match self
            .compute_relu_split_initial_bounds(
                graph,
                input,
                objective,
                threshold,
                engine,
                bootstrap_deadline,
                iterative_deadline,
            ) {
            Ok(bounds) => bounds,
            Err(NyError::DeadlineExceeded(_)) => {
                // The foundational bootstrap owns the full BaB deadline; if
                // that deadline is spent, the whole verifier timed out. The
                // root optimizer has a deliberately shorter warmup cap, whose
                // expiry leaves global search time but no root bounds and is
                // therefore an explicit warmup-cap Unknown (matching GPU BaB).
                let status = initial_bounds_deadline_status(Instant::now(), bootstrap_deadline);
                return Ok(match status {
                    BabVerificationStatus::Timeout => lifecycle.timeout_result(),
                    status => lifecycle.build_result(status),
                });
            }
            Err(error) => return Err(error),
        };
        // Graph-MIP stash (FIX 1): single-objective ReLU-split coverage — the
        // escalation reuses these per-property bounds instead of a truncated
        // recompute. Disabled when whole-network Graph-MIP is explicitly off
        // or the category requests no MIP reservation.
        crate::beta_crown::graph_mip_leaf::stash_root_bounds_for_mip(
            graph,
            input,
            &self.config.phase_budget,
            &initial_node_bounds,
        );

        info!(
            "Graph β-CROWN (ReLU split) initial objective: [{}, {}], threshold: {}, verify_upper={}",
            root_lower, root_upper, threshold, self.config.verify_upper_bound
        );

        // Quick verification/violation check.
        if let Some(early_result) =
            self.check_root_early_exit(root_lower, root_upper, threshold, &mut lifecycle)?
        {
            return Ok(early_result);
        }

        // Timeout guard: initial bounds computation (alpha-CROWN, CROWN-IBP)
        // may consume the entire budget for large models. Check before
        // entering the BaB loop (#2698). Use bab_timeout (not full timeout)
        // so post-BaB PGD reservation is respected (#4095).
        if lifecycle.start_time.elapsed() > bab_timeout {
            return Ok(lifecycle.timeout_result());
        }

        let graph_setup = build_graph_bab_setup(graph, &initial_node_bounds);

        // Create root domain and initialize alpha state.
        // When root α-CROWN optimization was run, transfer the optimized alpha
        // values to the root domain. This is the fix for #1851 Cause 1 — without
        // this, the SPSA/Adam-optimized alphas are discarded and replaced by the
        // `u > -l` heuristic, causing a massive bound quality gap.
        let mut root_domain = GraphBabDomain::root(
            initial_node_bounds,
            root_lower,
            root_upper,
            input,
            self.config.verify_upper_bound,
        )?;
        root_domain.alpha_state = build_root_alpha_state(
            graph,
            input,
            &root_domain.history,
            &graph_setup.initial_node_bounds_arc,
            root_alpha_state.as_ref(),
            self.config.beta_iterations > 0,
        );

        // Branch-and-bound queue
        let queue_priority = |lower: f32, upper: f32| self.config.violation_priority(lower, upper);
        let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
        root_domain.priority = queue_priority(root_domain.lower_bound, root_domain.upper_bound)?;
        queue.push(root_domain);

        // Identify ReLU nodes in the graph (sorted for deterministic branching order).
        // HashMap iteration order is randomized per-process; sorting ensures the
        // branching heuristic sees neurons in a stable order across runs.
        info!(
            "Found {} ReLU nodes for branching",
            graph_setup.relu_nodes.len()
        );

        let mut cut_pool = build_graph_cut_pool(
            graph,
            &graph_setup.initial_node_bounds_arc,
            &graph_setup.relu_nodes,
            &self.config,
        )?;
        let relu_nodes = graph_setup.relu_nodes;

        // Lambda optimization state (for cuts) — read from config (#2761)
        let mut lambda_opt_iter = 0usize;
        let lambda_opt_interval = self.config.lambda_opt_interval.max(1);
        let lambda_lr = self.config.lambda_lr;
        let lambda_beta1 = self.config.adaptive_config.beta1;
        let lambda_beta2 = self.config.adaptive_config.beta2;
        let lambda_epsilon = self.config.adaptive_config.epsilon;

        // Batch processing configuration
        let mut batch_size = self.config.batch_size.max(1);
        // Existing auto-enlarge presets keep their one-way legacy policy until
        // the independent exact-1 controller gate is also armed.
        let adaptive_enabled =
            adaptive_microbatch_controller_enabled(self.config.auto_enlarge_batch_size);
        let mut adaptive_microbatch = adaptive_enabled.then(|| {
            AdaptiveMicrobatchController::new(
                AdaptiveBatchRoute::GraphReluSplit,
                batch_size,
                queue.peek().map(estimate_graph_domain_bytes).unwrap_or(1),
                MicrobatchMemoryBudget::runtime(engine.is_some()),
            )
        });
        let mut batch_index = 0usize;

        while !queue.is_empty() {
            // Check termination conditions
            if let Some(termination) =
                self.check_termination(&mut lifecycle, &cut_pool, bab_timeout)
            {
                // #bab-frontier graph lane: export the surviving queue's
                // genuine subboxes (clip-shrunk domains) as attack seeds
                // before the queue is dropped (env-gated, guidance only —
                // see bab_frontier_export). Domains still covering the whole
                // root box are skipped inside the recorder.
                crate::beta_crown::bab_frontier_export::record_graph_bab_frontier_if_enabled(
                    queue.iter(),
                    input,
                );
                return Ok(termination);
            }

            let queue_batch_size = adaptive_microbatch
                .as_ref()
                .map_or(batch_size, |controller| {
                    controller.queue_pick_size(queue.len())
                });
            // Pop one queue batch.  In adaptive mode it stays owned here while
            // an independent device cursor can retry smaller prefixes.
            let mut batch: Vec<GraphBabDomain> = Vec::with_capacity(queue_batch_size);
            while batch.len() < queue_batch_size {
                if let Some(domain) = queue.pop() {
                    batch.push(domain);
                } else {
                    break;
                }
            }

            if batch.is_empty() {
                break;
            }

            // Pre-filter batch: check verified/violation/depth, generate cuts
            let domains_to_process =
                match self.pre_filter_batch(batch, threshold, &mut lifecycle, &mut cut_pool)? {
                    PreFilterOutcome::Violation => {
                        lifecycle.cuts_generated = cut_pool.total_generated;
                        return Ok(
                            lifecycle.build_result(BabVerificationStatus::PotentialViolation)
                        );
                    }
                    PreFilterOutcome::Process(domains) => domains,
                };

            if domains_to_process.is_empty() {
                continue;
            }

            // Lambda optimization: periodically optimize cut lambdas
            if self.config.enable_cuts
                && !cut_pool.is_empty()
                && lifecycle
                    .domains_explored
                    .is_multiple_of(lambda_opt_interval)
            {
                lambda_opt_iter += 1;

                if let Some(sample_domain) = domains_to_process.first() {
                    self.compute_graph_cut_gradients(
                        graph,
                        &mut cut_pool,
                        &sample_domain.node_bounds,
                        sample_domain.input_bounds.as_ref(),
                    );
                }

                for cut in cut_pool.cuts_mut() {
                    cut.update_lambda_adam(
                        lambda_lr,
                        lambda_beta1,
                        lambda_beta2,
                        lambda_epsilon,
                        lambda_opt_iter,
                    );
                }
                cut_pool.zero_grad();

                debug!(
                    "Lambda optimization iter {}: total_lambda = {:.4}",
                    lambda_opt_iter,
                    cut_pool.total_lambda()
                );
            }

            if let Some(controller) = adaptive_microbatch.as_mut() {
                let mut cursor = OrderedBatchCursor::new(domains_to_process.len());
                while !cursor.is_done() {
                    let requested = controller.current();
                    let range = cursor.next_range(requested);
                    let microbatch = &domains_to_process[range.clone()];
                    let split_depth = self.config.effective_relu_split_depth(microbatch.len());
                    let has_active_cuts = !cut_pool.is_empty() && self.config.enable_cuts;
                    let batch_start = Instant::now();
                    let batch_plan = GraphDomainBatchPlan::for_relu_split(
                        batch_index,
                        microbatch.len(),
                        requested,
                        engine.is_some(),
                        has_active_cuts,
                    );

                    let execution = match batch_plan.execution_mode() {
                        GraphDomainBatchExecutionMode::SharedExecutor => {
                            let domain_refs: Vec<&GraphBabDomain> = microbatch.iter().collect();
                            GraphDomainBatchExecutor::execute_single_objective(
                                self,
                                SingleObjectiveBatchRequest {
                                    graph,
                                    domains: &domain_refs,
                                    relu_nodes: &relu_nodes,
                                    objective,
                                    threshold,
                                    engine: engine.ok_or_else(|| {
                                        NyError::InvalidSpec(
                                            "shared relu-split executor requires a GemmEngine"
                                                .into(),
                                        )
                                    })?,
                                    cut_pool: None,
                                    retry_refusals: true,
                                },
                            )
                        }
                        GraphDomainBatchExecutionMode::ParallelFallback => Ok(microbatch
                            .par_iter()
                            .map(|domain| {
                                let _rayon_task_guard = RayonTaskGuard::new();
                                self.process_graph_domain_parallel(
                                    graph,
                                    domain,
                                    &relu_nodes,
                                    objective,
                                    threshold,
                                    None,
                                    split_depth,
                                )
                            })
                            .collect()),
                        GraphDomainBatchExecutionMode::SequentialFallback => Ok(self
                            .process_sequential_domains(
                                graph,
                                microbatch,
                                &relu_nodes,
                                objective,
                                threshold,
                                &mut cut_pool,
                                engine,
                                split_depth,
                            )?),
                    };

                    let results = match execution {
                        Ok(results) => results,
                        Err(reason) => match controller.on_refusal(reason) {
                            RefusalAction::Retry { previous, next } => {
                                info!(
                                    reason = reason.code(),
                                    previous_microbatch = previous,
                                    next_microbatch = next,
                                    queue_batch_size = domains_to_process.len(),
                                    retry_start = range.start,
                                    "Graph BaB: retrying refused microbatch"
                                );
                                continue;
                            }
                            RefusalAction::Exhausted => {
                                // No smaller batch exists.  Preserve the legacy
                                // sound sequential fallback for this one-domain
                                // range rather than losing the domain.
                                let domain_refs: Vec<&GraphBabDomain> = microbatch.iter().collect();
                                GraphDomainBatchExecutor::execute_single_objective(
                                    self,
                                    SingleObjectiveBatchRequest {
                                        graph,
                                        domains: &domain_refs,
                                        relu_nodes: &relu_nodes,
                                        objective,
                                        threshold,
                                        engine: engine.ok_or_else(|| {
                                            NyError::InvalidSpec(
                                                "shared relu-split executor requires a GemmEngine"
                                                    .into(),
                                            )
                                        })?,
                                        cut_pool: None,
                                        retry_refusals: false,
                                    },
                                )
                                .unwrap_or_else(|_| {
                                    unreachable!(
                                        "legacy graph batch execution never surfaces refusals"
                                    )
                                })
                            }
                        },
                    };

                    cursor.commit(range);
                    let observed_bytes = microbatch
                        .iter()
                        .map(estimate_graph_domain_bytes)
                        .sum::<usize>()
                        .div_ceil(microbatch.len().max(1));
                    controller.on_success(
                        requested,
                        microbatch.len(),
                        observed_bytes,
                        batch_start.elapsed(),
                        Some(
                            (lifecycle.start_time + bab_timeout)
                                .saturating_duration_since(Instant::now()),
                        ),
                    );

                    let queue_update_start = Instant::now();
                    if let Some(violation_result) = self.aggregate_bab_results(
                        results,
                        threshold,
                        &queue_priority,
                        &mut queue,
                        &mut lifecycle,
                        &mut cut_pool,
                    )? {
                        batch_plan.emit_to_sink(
                            self.graph_domain_batch_metrics_sink(),
                            GraphDomainBatchEmitTiming::new(batch_start.elapsed().as_secs_f64())
                                .with_queue_update(queue_update_start.elapsed().as_secs_f64()),
                        )?;
                        return Ok(violation_result);
                    }
                    batch_plan.emit_to_sink(
                        self.graph_domain_batch_metrics_sink(),
                        GraphDomainBatchEmitTiming::new(batch_start.elapsed().as_secs_f64())
                            .with_queue_update(queue_update_start.elapsed().as_secs_f64()),
                    )?;
                    batch_index += 1;
                }
                continue;
            }

            // Compute effective split depth for multi-depth ReLU splitting (#2767).
            // When the batch is smaller than the target, increase depth to generate
            // more children (2^k per domain instead of 2).
            // Reference: alpha-beta-CROWN `get_split_depth()` (bab.py:40-48).
            let split_depth = self
                .config
                .effective_relu_split_depth(domains_to_process.len());

            // Process domains: GPU-batched when engine available, parallel CPU otherwise
            let has_active_cuts = !cut_pool.is_empty() && self.config.enable_cuts;
            let batch_start = Instant::now();
            let batch_width = domains_to_process.len();
            let batch_plan = GraphDomainBatchPlan::for_relu_split(
                batch_index,
                batch_width,
                batch_size,
                engine.is_some(),
                has_active_cuts,
            );

            let results: Vec<GraphDomainResult> = match batch_plan.execution_mode() {
                GraphDomainBatchExecutionMode::SharedExecutor => {
                    let domain_refs: Vec<&GraphBabDomain> = domains_to_process.iter().collect();
                    GraphDomainBatchExecutor::execute_single_objective(
                        self,
                        SingleObjectiveBatchRequest {
                            graph,
                            domains: &domain_refs,
                            relu_nodes: &relu_nodes,
                            objective,
                            threshold,
                            engine: engine.ok_or_else(|| {
                                NyError::InvalidSpec(
                                    "shared relu-split executor requires a GemmEngine".into(),
                                )
                            })?,
                            cut_pool: None, // Single-objective path gates on !has_active_cuts
                            retry_refusals: false,
                        },
                    )
                    .unwrap_or_else(|_| {
                        unreachable!("legacy graph batch execution never surfaces refusals")
                    })
                }
                GraphDomainBatchExecutionMode::ParallelFallback => domains_to_process
                    .par_iter()
                    .map(|domain| {
                        let _rayon_task_guard = RayonTaskGuard::new();
                        self.process_graph_domain_parallel(
                            graph,
                            domain,
                            &relu_nodes,
                            objective,
                            threshold,
                            None,
                            split_depth,
                        )
                    })
                    .collect(),
                GraphDomainBatchExecutionMode::SequentialFallback => self
                    .process_sequential_domains(
                        graph,
                        &domains_to_process,
                        &relu_nodes,
                        objective,
                        threshold,
                        &mut cut_pool,
                        engine,
                        split_depth,
                    )?,
            };

            // Process results and enqueue children
            let queue_update_start = Instant::now();
            if let Some(violation_result) = self.aggregate_bab_results(
                results,
                threshold,
                &queue_priority,
                &mut queue,
                &mut lifecycle,
                &mut cut_pool,
            )? {
                batch_plan.emit_to_sink(
                    self.graph_domain_batch_metrics_sink(),
                    GraphDomainBatchEmitTiming::new(batch_start.elapsed().as_secs_f64())
                        .with_queue_update(queue_update_start.elapsed().as_secs_f64()),
                )?;
                return Ok(violation_result);
            }
            batch_plan.emit_to_sink(
                self.graph_domain_batch_metrics_sink(),
                GraphDomainBatchEmitTiming::new(batch_start.elapsed().as_secs_f64())
                    .with_queue_update(queue_update_start.elapsed().as_secs_f64()),
            )?;
            batch_index += 1;

            // Adaptive batch sizing (#4303): double when queue supplied a full batch.
            self.config.try_enlarge_batch_size(
                &mut batch_size,
                domains_to_process.len(),
                "Graph BaB",
            );
        }

        lifecycle.cuts_generated = cut_pool.total_generated;
        if lifecycle.has_unresolved() {
            return Ok(lifecycle.build_final_result());
        }

        info!(
            "Graph β-CROWN (ReLU split) verified after {} domains, {} verified, {} cuts",
            lifecycle.domains_explored, lifecycle.domains_verified, cut_pool.total_generated
        );

        Ok(lifecycle.build_final_result())
    }

    /// Compute initial node bounds and output bounds for the ReLU split BaB loop.
    ///
    /// Returns (node_bounds, optional_alpha_state, root_lower, root_upper).
    fn compute_relu_split_initial_bounds(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        bootstrap_deadline: Option<Instant>,
        iterative_deadline: Option<Instant>,
    ) -> Result<InitialBounds> {
        // #margin-subset-seed: publish the spec-referenced OUTPUT indices (the
        // objective's nonzero coefficient positions) for the scope of this
        // whole initial-bounds computation. The bootstrap's CROWN-IBP
        // collection consumes them at the OUTPUT node (crown_tighten.rs):
        // instead of the full `[output_dim x output_dim]` identity backward
        // (vggnet16: `[1000 x 401408]` conv buffers, 1.6 GiB each) it seeds
        // only the k referenced identity rows and scatters them over the
        // node's sound IBP bounds. Everything in this scope runs on this
        // thread; the RAII guard restores the empty publication on every exit
        // path, so BaB domains and other properties never observe it.
        let _margin_seed_guard =
            crate::output_margin_seed::MarginOutputSeedGuard::publish_from_objective(objective);

        // Step 1: Shared bootstrap — select alpha/IBP/CROWN-IBP mode and collect
        // intermediate node bounds. Delegates to the shared graph-BaB init service
        // (#1860 Packet C). Uses the global deadline: this foundational sweep is
        // mandatory and must reach every node (#4321).
        let mut bootstrap =
            compute_graph_bab_bootstrap(graph, input, &self.config, engine, bootstrap_deadline)?;

        // Spec-proven early-exit for the iterative root α-CROWN warmup (#warmup-early-exit).
        // The warmup always ran to the iteration / time cap even when the root objective
        // bound already cleared the decision threshold after a few iterations (cifar100
        // warmup overran its budget → 0 BaB domains). Carry the single objective +
        // threshold into the α-CROWN warmup loop via the alpha config so it can stop the
        // moment the projected root bound proves the property. SOUND: stops optimizing
        // sooner only — the exit-iteration bound is already a valid over-approximation
        // clearing the threshold. Only attached to the bootstrap's alpha_config, which
        // exclusively feeds the iterative `compute_graph_root_output_bounds` call below
        // (the mandatory node-bounds sweep already finished above).
        bootstrap.alpha_config.spec_early_exit = Some(crate::bounds::AlphaSpecEarlyExit {
            objective: objective.to_vec(),
            threshold,
            verify_upper_bound: self.config.verify_upper_bound,
        });
        // The bootstrap uses the full deadline because its foundational node
        // sweep must reach every node. Rebind the subsequent iterative root
        // alpha optimization to the shorter warmup fraction; the explicit
        // deadline argument below serves the non-alpha root paths.
        bootstrap.alpha_config.deadline = iterative_deadline;

        // Step 2: Compute initial output bounds via α-CROWN (if enabled) or CROWN.
        // This is single-objective-specific: we need a scalar (lower, upper) pair
        // from the objective vector, not the spec-matrix approach used by
        // multi-objective or GPU BaB.
        // Deadline check: fall back to IBP if deadline exceeded before expensive CROWN
        // backward pass. α-CROWN threads deadline via alpha_config already. (#3328)
        let initial_output = compute_graph_root_output_bounds(
            graph,
            input,
            &self.config,
            engine,
            &bootstrap,
            iterative_deadline,
        )?;
        let (root_lower, root_upper) = objective_bounds(&initial_output, objective)?;

        Ok((
            bootstrap.initial_node_bounds,
            bootstrap.root_alpha_state,
            root_lower,
            root_upper,
        ))
    }
}

#[cfg(test)]
mod deadline_status_tests {
    use super::*;

    #[test]
    fn spent_bootstrap_deadline_is_global_timeout() {
        let now = Instant::now();
        let spent = now
            .checked_sub(std::time::Duration::from_millis(1))
            .expect("system uptime exceeds one millisecond");

        assert_eq!(
            initial_bounds_deadline_status(now, Some(spent)),
            BabVerificationStatus::Timeout
        );
    }

    #[test]
    fn live_bootstrap_deadline_is_internal_warmup_unknown() {
        let now = Instant::now();
        let live = now + std::time::Duration::from_secs(1);

        assert!(matches!(
            initial_bounds_deadline_status(now, Some(live)),
            BabVerificationStatus::Unknown { reason }
                if reason.contains("Initial-bound warmup exceeded its deadline cap")
        ));
    }
}
