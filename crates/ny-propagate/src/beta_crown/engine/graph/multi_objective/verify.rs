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
use std::time::Instant;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::instrument;

use crate::beta_crown::result::BetaCrownResult;
use crate::layers::Layer;
use crate::GraphNetwork;

use super::super::super::BetaCrownVerifier;
use super::super::domain_batch::{
    GraphDomainBatchEmitTiming, GraphDomainBatchExecutionMode, GraphDomainBatchExecutor,
    GraphDomainBatchPlan, MultiObjectiveBatchRequest,
};
use super::super::shared::state::GraphBabLifecycle;
use super::finalize::finalize_multi_objective_result;
use super::queue::{apply_batched_results, pop_domain_batch, prefilter_batch, LeafOracleCtx};
use super::root::{
    evaluate_root, validate_multi_objective_inputs, MultiObjectiveRootOutcome,
    MultiObjectiveRootRequest, MultiObjectiveRootState,
};
use super::sequential::SequentialMultiObjectiveContext;

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
            graph, input, objectives, thresholds, engine,
            false, // conjunctive=false → disjunctive semantics
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
            graph, input, objectives, thresholds, engine,
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
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: Option<&dyn GemmEngine>,
        conjunctive: bool,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        let graph = self.configured_graph_for_crown(graph);
        let graph = &graph;
        let now = Instant::now();
        let mut lifecycle = GraphBabLifecycle::new(now);
        validate_multi_objective_inputs(objectives, thresholds)?;
        let root = match evaluate_root(
            MultiObjectiveRootRequest {
                verifier: self,
                graph,
                input,
                objectives,
                thresholds,
                engine,
                conjunctive,
                deadline,
            },
            &mut lifecycle,
        )? {
            MultiObjectiveRootOutcome::Finished(result) => return Ok(*result),
            MultiObjectiveRootOutcome::Continue(root) => *root,
        };
        let MultiObjectiveRootState {
            root_domain,
            relu_nodes,
            mut cut_pool,
            use_batched_gpu,
        } = root;

        // BaB timeout with post-BaB PGD reservation (#4095).
        // When a wall-clock deadline is provided (#4321), derive the effective
        // timeout from remaining time instead of the configured timeout.
        let pgd_frac = self
            .config
            .phase_budget
            .post_bab_pgd_fraction
            .clamp(0.0, 0.5);
        // #cora-double-reserve: a Some(deadline) is the CLI ledger's bab_deadline,
        // which ALREADY reserved post_bab_pgd_fraction once (phase_budget.rs
        // bab_deadline). Scaling it again here double-applied the fraction and
        // silently burned ~10-16% of the internal tier on every deadline-threaded
        // multi-objective run. Apply the fraction only when self-budgeting (None).
        let bab_timeout = match deadline {
            Some(dl) => dl.saturating_duration_since(now),
            None => self.config.timeout.mul_f32(1.0 - pgd_frac),
        };

        let batch_size = self.config.batch_size.max(1);

        // Conflict-clause learning, graph port (win-plan arc C, v2): per-run
        // store, gated NY_BAB_CLAUSE_LEARN=1 (default OFF => disabled store =>
        // byte-identical loop). Scope of THIS store: one graph, one root input
        // box, one (objectives, thresholds, conjunctive) tuple — a clause is
        // only ever recorded from a domain closed verified under the SAME
        // objective semantics it prunes for (see `prefilter_batch` /
        // `apply_batched_results` for the per-close argument, and
        // `conflict_clauses_graph` for the region-inclusion + purity-guard
        // argument). The sequential (non-batched) child path is deliberately
        // NOT a record site in v2 — fail-safe: fewer clauses, never unsound.
        let mut clause_store =
            crate::beta_crown::conflict_clauses_graph::GraphClauseStore::from_env();
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
        let looseness_probe = std::env::var("NY_LOOSENESS_PROBE").ok().as_deref() == Some("1");
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
                        )
                    );
                    last_bab_trace = Some(Instant::now());
                    waves_since_bab_trace = 0;
                }
            }
            // #complete-before-deadline: recognize a DRAINED queue before the
            // wall-clock deadline check. When the final batch's processing pushes
            // `elapsed` past `bab_timeout` AND verifies the last open domains, the
            // queue is now empty — a COMPLETE, sound proof. The deadline is only a
            // budget cap; finishing at/just-after it does not invalidate the proof.
            // Without this, `check_termination` below would fire on this iteration
            // and discard the completed verification as a timeout — which
            // disproportionately loses instances that converge right at the budget.
            // Soundness is unchanged: the break falls through to
            // `finalize_multi_objective_result`, which returns `Verified` ONLY when
            // `domains_verified > 0 && queue.is_empty() && !has_unresolved()` — a
            // violated/evicted/no-branch/propagation-failure domain still yields
            // `Unknown`, never a false `Verified`.
            if queue.is_empty() {
                break;
            }
            if let Some(result) = lifecycle.check_termination(bab_timeout, self.config.max_domains)
            {
                if prof {
                    eprintln!(
                        "[NY_ACASXU_PROF] terminate(timeout/limit): explored={} verified={} queue={} max_depth={} elapsed={:.2}s",
                        lifecycle.domains_explored,
                        lifecycle.domains_verified,
                        queue.len(),
                        lifecycle.max_depth_reached,
                        lifecycle.start_time.elapsed().as_secs_f64(),
                    );
                }
                return Ok(result);
            }

            let batch = pop_domain_batch(&mut queue, batch_size);
            if batch.is_empty() {
                break;
            }
            // #endgame-grace: this pop took the ENTIRE remaining frontier — the
            // batched executor may then finish its chunks within the bounded
            // NY_ENDGAME_GRACE_SECS overrun instead of deadline-dropping them
            // (a dropped tail turns a fully-verifying tree into Unknown).
            let endgame = queue.is_empty();

            // NY_UNSTABLE_COUNT (dark, diagnostic-only): for the frontier-WORST subdomain
            // count unstable ReLU pre-activation neurons (l<0<u) = the # binaries an exact
            // MILP leaf-verifier would face. Cheap (no sampling). Prints once per new-min
            // margin so we see the count at the plateau depth. Print-only, sound.
            if std::env::var("NY_UNSTABLE_COUNT").ok().as_deref() == Some("1") {
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
                self.config.enable_cuts,
                &mut lifecycle,
                &mut cut_pool,
                &mut clause_store,
            )?;

            if domains_to_process.is_empty() {
                continue;
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
                let cut_pool_ref = if self.config.enable_cuts && !cut_pool.is_empty() {
                    Some(&cut_pool)
                } else {
                    None
                };
                let domain_refs: Vec<_> = domains_to_process.iter().collect();

                let results = GraphDomainBatchExecutor::execute_multi_objective(
                    self,
                    MultiObjectiveBatchRequest {
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
                        endgame,
                    },
                );

                let queue_update_start = Instant::now();
                // Graph-MIP LEAF oracle context (increment 6): only built when
                // the default-on CLI attached an oracle; `None` under the
                // exact-zero kill switch keeps the requeue byte-identical.
                let leaf_ctx = self.graph_mip_leaf_oracle().map(|oracle| LeafOracleCtx {
                    oracle,
                    graph,
                    objectives,
                    thresholds,
                    deadline,
                });
                apply_batched_results(
                    results,
                    &mut queue,
                    &mut cut_pool,
                    &mut lifecycle,
                    self.config.enable_cuts,
                    leaf_ctx.as_ref(),
                    &mut clause_store,
                )?;
                batch_plan.emit_to_sink(
                    self.graph_domain_batch_metrics_sink(),
                    GraphDomainBatchEmitTiming::new(batch_start.elapsed().as_secs_f64())
                        .with_queue_update(queue_update_start.elapsed().as_secs_f64()),
                )?;
                batch_index += 1;
                continue;
            }

            self.process_multi_objective_domains_sequential(SequentialMultiObjectiveContext {
                graph,
                domains_to_process,
                relu_nodes: &relu_nodes,
                objectives,
                thresholds,
                engine,
                cut_pool: &mut cut_pool,
                queue: &mut queue,
                domains_verified: &mut lifecycle.domains_verified,
                unresolved_due_to_no_branch: &mut lifecycle.unresolved_due_to_no_branch,
                unresolved_due_to_violated_drop: &mut lifecycle.unresolved_due_to_violated_drop,
                unresolved_due_to_propagation_failure: &mut lifecycle
                    .unresolved_due_to_propagation_failure,
                conjunctive,
                deadline: Some(lifecycle.start_time + bab_timeout),
            })?;
            batch_plan.emit_to_sink(
                self.graph_domain_batch_metrics_sink(),
                GraphDomainBatchEmitTiming::new(batch_start.elapsed().as_secs_f64()),
            )?;
            batch_index += 1;
        }

        lifecycle.cuts_generated = cut_pool.total_generated;
        if clause_store.is_enabled() {
            tracing::debug!(
                clause_pruned = clause_store.pruned_count(),
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
/// solver. `worst_lb` is `None` when every open domain is fully verified — rendered
/// as `all-verified` so the field is never blank.
pub(crate) fn format_bab_trace_line(
    wall_elapsed_secs: f64,
    worst_lb: Option<f32>,
    open_domains: usize,
    max_depth: usize,
    domains_processed: usize,
) -> String {
    let worst = match worst_lb {
        Some(w) => format!("{w:.6}"),
        None => "all-verified".to_string(),
    };
    format!(
        "[bab-trace] t={wall_elapsed_secs:.3}s worst_unverified_lb={worst} \
         open_domains={open_domains} max_depth={max_depth} domains_processed={domains_processed}"
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
        let line = format_bab_trace_line(12.5, Some(-0.375), 42, 7, 1234);
        assert_eq!(
            line,
            "[bab-trace] t=12.500s worst_unverified_lb=-0.375000 \
             open_domains=42 max_depth=7 domains_processed=1234"
        );
    }

    /// The `all-verified` sentinel renders when no objective is left open.
    #[test]
    fn trace_line_all_verified_sentinel() {
        let line = format_bab_trace_line(0.0, None, 0, 3, 99);
        assert_eq!(
            line,
            "[bab-trace] t=0.000s worst_unverified_lb=all-verified \
             open_domains=0 max_depth=3 domains_processed=99"
        );
    }
}
