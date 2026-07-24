// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! GPU-accelerated branch-and-bound using DomainList.
//!
//! This module provides a BaB loop that uses `DomainList` for domain storage
//! instead of `BinaryHeap<GraphBabDomain>`. This enables efficient GPU batch
//! processing with the `pick_out_batched()` pattern.
//!
//! # Architecture
//!
//! ```text
//! DomainList (CPU storage)
//!   ├── layer_bounds: HashMap<String, TensorStorage>
//!   ├── input_bounds: TensorStorage
//!   ├── global_lbs/ubs: TensorStorage
//!   └── metadata: Vec<DomainMetadata>
//!           ↓ pick_out_batched()
//! PickedDomains (zero-copy extraction)
//!           ↓ GPU propagation
//! ProcessedDomains
//!           ↓ add()
//! DomainList (updated)
//! ```
//!
//! # Module Structure
//!
//! - `batched_gpu` — GPU batched execution (branching, backward pass, beta refinement)
//! - `check` — Domain verification/violation checks and result construction
//! - `cpu_fallback` — CPU parallel evaluation fallback (no GPU engine)
//! - `init` — Root bound computation and DomainList setup
//! - `input_split` — Input-split branching path
//! - `prefilter` — Pre-filtering of picked domains before branching
//!
//! # Reference
//!
//! - Design: `designs/2026-02-02-gpu-bab-execution-path.md`
//! - Decomposition: `designs/2026-02-10-code-structure-wave8-domain-list-gpu-bab-split.md`
//! - Alpha-beta-CROWN: `complete_verifier/branching_domains.py`:87-412

mod batched_gpu;
pub(crate) mod check;
mod cpu_fallback;
mod grouped_disjunctive;
mod init;
mod input_split;
mod input_split_support;
mod kfsb;
mod metrics;
mod parent_contexts;
pub(crate) mod prefilter;

use std::time::Instant;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::batched_domain::BatchedDomainOptions;
use crate::beta_crown::branching::BranchingHeuristic;
use crate::beta_crown::engine::graph::shared::setup::build_graph_bab_setup;
use crate::beta_crown::result::BabVerificationStatus;
use crate::GraphNetwork;

use super::super::super::BetaCrownVerifier;
use super::adaptive_microbatch::{
    adaptive_microbatch_controller_enabled, estimate_domain_list_bytes_per_domain,
    estimate_picked_bytes_per_domain, AdaptiveBatchRoute, AdaptiveMicrobatchController,
    MicrobatchMemoryBudget, MicrobatchRefusalReason, OrderedBatchCursor, RefusalAction,
};
use super::domain_conversion::processed_from_graph_domains_with_la;
use super::input_split::shared::graph_spec_ibp_root_screen_with_deadline;

use batched_gpu::{GpuBatchContext, GpuBatchOutcome};
use check::{check_domain_bounds, BabLoopState, DomainCheckResult};
use cpu_fallback::{process_cpu_fallback_batch, CpuFallbackOutcome};
use init::{build_setup_context, compute_initial_bounds, create_domain_list};
use input_split::{
    process_input_split_batch, process_input_split_batch_attempt, InputSplitOutcome,
};
use prefilter::prefilter_picked_domains;

impl BetaCrownVerifier {
    /// Verify GraphNetwork using GPU-accelerated BaB with DomainList storage.
    ///
    /// This is an alternative to `verify_graph_relu_split` that uses `DomainList`
    /// instead of `BinaryHeap<GraphBabDomain>`. Benefits:
    ///
    /// - Batched tensor storage for efficient GPU transfer
    /// - `pick_out_batched()` yields GPU-ready `BatchedDomains`
    /// - Reduces per-domain overhead for large domain counts
    ///
    /// # Arguments
    /// * `graph` - The graph network to verify
    /// * `input` - Input bounds specification
    /// * `objective` - Objective vector for linear combination
    /// * `threshold` - Verification threshold
    /// * `engine` - Optional GPU engine for acceleration
    ///
    /// # Reference
    /// Design: `designs/2026-02-02-gpu-bab-execution-path.md`
    /// `deadline`: If `Some`, the BaB engine derives its phase budgets from
    /// remaining wall-clock time instead of `self.config.timeout` (#4321).
    pub fn verify_graph_gpu_domain_list(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<crate::beta_crown::result::BetaCrownResult> {
        let graph = self.configured_graph_for_crown(graph);
        let graph = &graph;
        let engine = self.resolve_engine(engine);
        let now = Instant::now();
        let mut state = BabLoopState::new(now);
        let is_input_split_mode = matches!(
            &self.config.branching_heuristic,
            BranchingHeuristic::InputSplit
        );
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
        // Foundational IBP pre-screen + node-bounds bootstrap must reach every
        // node, so they get the full global deadline (not the warmup fraction);
        // capping it choked conv-heavy DAGs into premature "deadline exceeded
        // before node 'Conv_0'" with budget unused (#4321).
        let initial_deadline = Some(now + bab_timeout);
        // The genuinely-iterative root α-CROWN warmup + spec-guided output
        // optimization is capped at `initial_bounds_fraction` of the BaB budget
        // (#4095/#4413), mirroring the CPU ReLU-split path. With fraction 0.0 this
        // expires instantly so the alpha warmup bails to a warmup-cap Unknown
        // instead of consuming the whole budget before branching.
        let iterative_deadline = {
            let frac = self
                .config
                .phase_budget
                .initial_bounds_fraction
                .clamp(0.0, 1.0);
            Some(now + bab_timeout.mul_f32(frac))
        };

        // IBP root pre-screen (Part of #3870, parity with CPU single_objective.rs:156-176).
        //
        // When input_split_ibp_enhancement is enabled, run a cheap IBP pass on the
        // root domain before the expensive alpha-CROWN / CROWN bootstrap. If IBP
        // alone verifies the root, skip all warmup computation entirely.
        if is_input_split_mode && self.config.input_split_ibp_enhancement {
            let spec_matrix =
                ndarray::Array2::from_shape_vec((1, objective.len()), objective.to_vec()).map_err(
                    |e| ny_core::NyError::InvalidSpec(format!("IBP pre-screen spec matrix: {}", e)),
                )?;
            match graph_spec_ibp_root_screen_with_deadline(
                graph,
                input,
                &spec_matrix,
                engine,
                initial_deadline,
            ) {
                Ok((ibp_bounds, _)) => {
                    let ibp_lower = ibp_bounds.lower()[[0]];
                    let ibp_upper = ibp_bounds.upper()[[0]];
                    if self
                        .config
                        .domain_is_verified(ibp_lower, ibp_upper, threshold)
                    {
                        info!(
                            "GPU BaB: root domain verified by IBP alone \
                             (lower={}, upper={}, threshold={})",
                            ibp_lower, ibp_upper, threshold
                        );
                        state.domains_explored = 1;
                        state.domains_verified = 1;
                        let output_bounds = BoundedTensor::new(
                            ndarray::arr1(&[ibp_lower]).into_dyn(),
                            ndarray::arr1(&[ibp_upper]).into_dyn(),
                        )?;
                        return Ok(state.build_result_with_bounds(
                            BabVerificationStatus::Verified,
                            output_bounds,
                        ));
                    }
                }
                Err(ny_core::NyError::DeadlineExceeded(_)) => {
                    info!(
                        "GPU BaB: skipping root IBP pre-screen because the warmup deadline expired"
                    );
                }
                Err(err) => return Err(err),
            }
        }

        // Step 1-2: Compute initial bounds (alpha-CROWN, IBP, or CROWN-IBP)
        // Thread deadline so α-CROWN bails early if timeout budget exhausted (#2698)
        //
        // Warmup cap (#2206 Packet C, #4095): initial bounds get at most
        // `initial_bounds_fraction` of the BaB timeout. Mirrors core.rs pattern.
        let mut init_result = match compute_initial_bounds(
            graph,
            input,
            objective,
            &self.config,
            engine,
            initial_deadline,
            iterative_deadline,
            is_input_split_mode,
        ) {
            Ok(init) => init,
            Err(ny_core::NyError::DeadlineExceeded(_)) => {
                info!(
                    "GPU BaB: initial-bound warmup exceeded its deadline cap after {:.3}s; returning Unknown",
                    state.start_time.elapsed().as_secs_f64()
                );
                return Ok(state.build_result(BabVerificationStatus::Unknown {
                    reason: "Initial-bound warmup exceeded its deadline cap before branching"
                        .to_string(),
                }));
            }
            Err(err) => return Err(err),
        };

        // Fix (#4095): compute_initial_bounds stores the warmup deadline in
        // InputSplitBootstrap.deadline, which then leaks into per-domain BaB
        // CROWN passes. Per-domain passes should use the full bab_timeout,
        // not the warmup fraction.
        if let Some(ref mut bootstrap) = init_result.input_split_bootstrap {
            bootstrap.deadline = Some(state.start_time + bab_timeout);
        }

        info!(
            "GPU BaB (DomainList): initial bounds [{:.4}, {:.4}], threshold: {:.4}",
            init_result.root_lower, init_result.root_upper, threshold
        );

        // Check immediate verification or violation at root
        state.domains_explored = 1;
        match check_domain_bounds(
            init_result.root_lower,
            init_result.root_upper,
            threshold,
            self.config.verify_upper_bound,
        ) {
            DomainCheckResult::Verified => {
                state.domains_verified = 1;
                return Ok(state.build_result_with_bounds(
                    BabVerificationStatus::Verified,
                    init_result.initial_output,
                ));
            }
            DomainCheckResult::Violation => {
                return Ok(state.build_result_with_bounds(
                    BabVerificationStatus::PotentialViolation,
                    init_result.initial_output,
                ));
            }
            DomainCheckResult::Undecided => {
                // Reset — root is explored but not yet verified
                state.domains_explored = 0;
                state.domains_verified = 0;
            }
        }

        // Timeout guard: initial bounds computation (alpha-CROWN, IBP) may
        // consume the entire budget for large models. Check before entering
        // the BaB loop (#2698). Use bab_timeout for PGD reservation (#4095).
        if let Some(result) = state.check_termination(bab_timeout, self.config.max_domains) {
            return Ok(result);
        }

        // Step 3: Initialize DomainList with root domain
        let graph_setup = build_graph_bab_setup(graph, &init_result.initial_node_bounds);
        let (mut domain_list, layer_names) = create_domain_list(
            &init_result,
            input,
            graph,
            &self.config,
            is_input_split_mode,
            &graph_setup,
        )?;
        let setup = build_setup_context(graph, &self.config, graph_setup.relu_nodes);

        // Step 4: BaB loop using DomainList
        //
        // For input-split mode, cap the pick batch size to control termination
        // granularity and scheduling overhead. With rayon parallelism in
        // compute_crown_or_ibp_bounds_batched (Phase 2, #3870), the per-domain
        // CROWN cost is amortized across CPU cores, so the cap is no longer
        // a throughput bottleneck — it just bounds how many domains are processed
        // between termination checks.
        let mut batch_size = if is_input_split_mode {
            use super::input_split::batching::input_split_loop_batch_size;
            input_split_loop_batch_size(self.config.batch_size, input.len())?.effective_batch_size
        } else {
            self.config.batch_size.max(1)
        };
        // Both opt-ins are required. If the independent controller gate is
        // dark, even an auto-enlarge preset retains its historical route.
        let adaptive_enabled = is_input_split_mode
            && adaptive_microbatch_controller_enabled(self.config.auto_enlarge_batch_size);
        let mut adaptive_microbatch = adaptive_enabled.then(|| {
            AdaptiveMicrobatchController::new(
                AdaptiveBatchRoute::DomainListInputSplit,
                batch_size,
                estimate_domain_list_bytes_per_domain(&domain_list),
                MicrobatchMemoryBudget::runtime(engine.is_some()),
            )
        });
        let batch_options = BatchedDomainOptions {
            enable_interm_transfer: self.config.enable_interm_transfer,
        };

        // Periodic domain re-sorting: reorder DomainList by the same queue
        // priority the CPU heap uses, so pick_out_batched extracts the same
        // frontier ordering after each sort interval. Matches
        // alpha-beta-CROWN's sort_domain_interval behavior (bab.py:168).
        //
        // The DomainList uses BreadthFirst (FIFO/queue) traversal so that
        // after a priority sort, the highest-priority domains at the front
        // of the queue are processed before newly-split children (which are
        // appended at the back). This preserves the sorted CPU-equivalent
        // ordering between sort intervals more closely than DepthFirst/LIFO,
        // which would process new children before the sorted frontier.
        //
        // Input-split mode uses interval=1 (sort before every pick) for
        // best-first ordering parity with the CPU BinaryHeap. Without this,
        // newly-split children with high priority sit at the back of the
        // FIFO queue for multiple iterations, causing the GPU path to waste
        // computation on low-priority domains. Benchmark evidence (#3870):
        // 2.3% verification rate at interval=3 vs 22% on CPU heap.
        //
        // ReLU-split mode keeps interval=3 since the GPU batch backward
        // pass dominates iteration cost and the sort overhead is less
        // amortized.
        let sort_interval = if is_input_split_mode { 1 } else { 3 };
        let mut iterations_since_sort = 0usize;
        let mut batch_index = 0usize;

        while !domain_list.is_empty() {
            // Check termination conditions (use bab_timeout for PGD reservation #4095)
            if let Some(result) = state.check_termination(bab_timeout, self.config.max_domains) {
                // #bab-frontier graph lane: the surviving DomainList is exactly
                // where a CE must live; export the top-K subboxes as attack
                // seeds before it is dropped (env-gated, guidance only — see
                // bab_frontier_export). Domains without their own subbox
                // (pure ReLU-split rows still covering the root box) are
                // skipped inside the recorder.
                crate::beta_crown::bab_frontier_export::record_domain_list_frontier_if_enabled(
                    &domain_list,
                    input,
                    self.config.verify_upper_bound,
                );
                return Ok(result);
            }

            // Periodic domain re-sorting: keep the DomainList frontier aligned
            // with CPU BaB's queue priority before the next pick_out_batched.
            iterations_since_sort += 1;
            let queue_batch_size = adaptive_microbatch
                .as_ref()
                .map_or(batch_size, |controller| {
                    controller.queue_pick_size(domain_list.len())
                });
            if iterations_since_sort >= sort_interval && domain_list.len() > queue_batch_size {
                domain_list.sort_by_domain_priority(self.config.verify_upper_bound)?;
                iterations_since_sort = 0;
            }

            // Queue batch and device microbatch are separate when adaptation is
            // enabled.  The picked queue batch remains intact while the ordered
            // cursor retries smaller device index ranges after a refusal.
            let picked = domain_list.pick_out_batched(queue_batch_size, batch_options)?;

            if picked.batch_size == 0 {
                break;
            }

            if is_input_split_mode {
                let input_split_bootstrap =
                    init_result.input_split_bootstrap.as_ref().ok_or_else(|| {
                        ny_core::NyError::InternalError(
                            "GPU BaB input split: missing root bootstrap context".to_string(),
                        )
                    })?;
                if let Some(controller) = adaptive_microbatch.as_mut() {
                    let mut cursor = OrderedBatchCursor::new(picked.batch_size);
                    while !cursor.is_done() {
                        let requested = controller.current();
                        let range = cursor.next_range(requested);
                        // Keep the queue batch in place and address the device
                        // microbatch by ordered indices. This avoids copying its
                        // tensors and makes a refusal retry the same prefix.
                        let processable_picked_indices: Vec<usize> = range.clone().collect();
                        let mut use_host_fallback = false;
                        let committed = loop {
                            let attempt_start = Instant::now();
                            let attempt_engine = if use_host_fallback { None } else { engine };
                            match process_input_split_batch_attempt(
                                self,
                                graph,
                                &picked,
                                &processable_picked_indices,
                                objective,
                                input_split_bootstrap,
                                threshold,
                                attempt_engine,
                                &state,
                                batch_index,
                            ) {
                                Ok(effects) => {
                                    let outcome = effects.commit(&mut state, &mut domain_list)?;
                                    let elapsed = attempt_start.elapsed();
                                    let observed_bytes = estimate_picked_bytes_per_domain(&picked);
                                    cursor.commit(range.clone());
                                    controller.on_success(
                                        requested,
                                        processable_picked_indices.len(),
                                        observed_bytes,
                                        elapsed,
                                        Some(
                                            (state.start_time + bab_timeout)
                                                .saturating_duration_since(Instant::now()),
                                        ),
                                    );
                                    batch_index += 1;
                                    if matches!(outcome, InputSplitOutcome::Violation) {
                                        return Ok(state.build_result(
                                            BabVerificationStatus::PotentialViolation,
                                        ));
                                    }
                                    break true;
                                }
                                Err(error) => {
                                    let Some(reason) = MicrobatchRefusalReason::from_error(&error)
                                    else {
                                        return Err(error);
                                    };
                                    match controller.on_refusal(reason) {
                                        RefusalAction::Retry { previous, next } => {
                                            info!(
                                                reason = reason.code(),
                                                previous_microbatch = previous,
                                                next_microbatch = next,
                                                queue_batch_size = picked.batch_size,
                                                retry_start = range.start,
                                                "GPU BaB input split: retrying refused microbatch"
                                            );
                                            break false;
                                        }
                                        RefusalAction::Exhausted
                                            if engine.is_some() && !use_host_fallback =>
                                        {
                                            use_host_fallback = true;
                                            info!(
                                                reason = reason.code(),
                                                queue_batch_size = picked.batch_size,
                                                retry_start = range.start,
                                                "GPU BaB input split: retrying one-domain \
                                                 refusal on host"
                                            );
                                            continue;
                                        }
                                        RefusalAction::Exhausted => return Err(error),
                                    }
                                }
                            }
                        };
                        if !committed {
                            // Refusal backoff did not advance the cursor; the
                            // next iteration addresses the same ordered prefix.
                            continue;
                        }
                    }
                    continue;
                } else {
                    // Exact legacy path unless both controller opt-ins hold.
                    let processable_picked_indices: Vec<usize> = (0..picked.batch_size).collect();
                    match process_input_split_batch(
                        self,
                        graph,
                        &picked,
                        &processable_picked_indices,
                        objective,
                        input_split_bootstrap,
                        threshold,
                        engine,
                        &mut state,
                        &mut domain_list,
                        batch_index,
                    )? {
                        InputSplitOutcome::Continue => {
                            batch_index += 1;
                            continue;
                        }
                        InputSplitOutcome::Violation => {
                            return Ok(
                                state.build_result(BabVerificationStatus::PotentialViolation)
                            );
                        }
                    }
                }
            }

            state.domains_explored += picked.batch_size;

            // Compute batched unstable neurons and branching decisions BEFORE domain
            // materialization so the ReLU fast path can branch directly from PickedDomains.
            // GenBaB uses find_splittable_graph_nodes instead, so skip batched computation.
            // InputSplit doesn't use ReLU unstable neurons at all — it branches on input dimensions.
            let is_genbab = matches!(
                &self.config.branching_heuristic,
                BranchingHeuristic::GenBaB(_)
            );
            let (unstable_batched, branches_batched) = if !is_genbab && !is_input_split_mode {
                let unstable = picked.find_unstable_neurons_batched(&setup.relu_pre_map)?;
                let branches = picked.select_branch_batched(&unstable, &setup.relu_pre_map)?;
                (unstable, branches)
            } else {
                (Vec::new(), Vec::new())
            };

            // Pre-filter picked domains: separate verified/violated/max-depth from processable
            let filter_result = prefilter_picked_domains(
                &picked.metadata,
                threshold,
                self.config.verify_upper_bound,
                self.config.max_depth,
                &mut state,
            );

            if filter_result.violation {
                return Ok(state.build_result(BabVerificationStatus::PotentialViolation));
            }

            let processable_picked_indices = filter_result.processable_indices;
            if processable_picked_indices.is_empty() {
                continue;
            }

            // GPU or CPU execution path
            if let Some(eng) = engine {
                // GPU path: branch, batch, evaluate. Extracted to batched_gpu.rs.
                let gpu_ctx = GpuBatchContext {
                    graph,
                    eng,
                    picked: &picked,
                    processable_picked_indices: &processable_picked_indices,
                    unstable_batched: &unstable_batched,
                    branches_batched: &branches_batched,
                    is_genbab,
                    setup: &setup,
                    objective,
                    threshold,
                    layer_names: &layer_names,
                };
                match self.process_gpu_batched(&gpu_ctx, &mut state, &mut domain_list)? {
                    GpuBatchOutcome::Violation => {
                        return Ok(state.build_result(BabVerificationStatus::PotentialViolation));
                    }
                    GpuBatchOutcome::Continue(_captured_la) => {}
                }
            } else {
                // CPU fallback: materialize and process in parallel. Extracted to cpu_fallback.rs.
                match process_cpu_fallback_batch(
                    self,
                    graph,
                    &picked,
                    &processable_picked_indices,
                    &layer_names,
                    &setup.relu_nodes,
                    objective,
                    threshold,
                    &mut state,
                )? {
                    CpuFallbackOutcome::Violation => {
                        return Ok(state.build_result(BabVerificationStatus::PotentialViolation));
                    }
                    CpuFallbackOutcome::Children(children) => {
                        if !children.is_empty() {
                            let processed = processed_from_graph_domains_with_la(
                                &children,
                                &layer_names,
                                self.config.enable_interm_transfer,
                                None,
                            )?;
                            domain_list.add(processed)?;
                        }
                    }
                }
            }

            // Adaptive batch sizing (#4303): double when domain list supplied a full batch.
            self.config
                .try_enlarge_batch_size(&mut batch_size, picked.batch_size, "GPU BaB");

            info!(
                "GPU BaB iteration: explored={}, verified={}, batch_size={}",
                state.domains_explored, state.domains_verified, picked.batch_size
            );
        }

        // Queue exhaustion with no unresolved flags → all domains verified.
        // Mirrors relu_split.rs termination: when the BaB loop exits normally
        // (domain_list empty, no timeout, no unresolved flags), every domain was
        // either verified inline or its children were verified recursively.
        //
        // Previous bug (#1896): used `domains_verified == domains_explored` which
        // can never be true when branching occurs — parent domains increment
        // `domains_explored` but only their children increment `domains_verified`,
        // and these are disjoint sets.
        //
        // Ref: alpha-beta-CROWN bab.py:general_bab returns 'safe' when the domain
        // list is exhausted without setting any other result.
        //
        // Queue-cap eviction (max_queue_size) deletes unverified domains, so a
        // drained queue after any eviction covers only part of the search
        // space — the exhaustion argument above no longer holds and the result
        // must be Unknown.
        if domain_list.evicted_count() > 0 {
            state.unresolved_due_to_eviction = true;
        }
        Ok(state.build_final_result())
    }
}

#[cfg(test)]
mod tests;
