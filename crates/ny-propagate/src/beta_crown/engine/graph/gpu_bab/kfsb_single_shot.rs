// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #kfsb-single-shot: αβ-CROWN-style candidate scoring for the graph kFSB
//! selector (dark, typed `bab.kfsb_single_shot`).
//!
//! MEASURED MOTIVATION (TinyYOLO prop_000030, yolo_2023, 300 s, pin
//! `ny_eng9_afca3d021`, `-vv` timeline): the conjunctive multi-objective
//! sequential lane spends ~34 s per BaB domain, two thirds of it INSIDE branch
//! selection (~10 s prescore + ~12.4 s candidate simulations; the committed
//! children's own bounding is the other ~11 s), because the
//! aggregation-critical kFSB route (`select_graph_branch_multi_kfsb` ->
//! [`BetaCrownVerifier::select_graph_branch_kfsb_in_gpu_batched`]) pays for
//! two stand-ins that no other kFSB implementation uses:
//!
//! 1. the BaBSR PRESCORE is seeded with the IDENTITY over the whole graph
//!    output (`compute_graph_babsr_scores` → `objective_seed: None`): on
//!    TinyYOLO that is a 21125 x 21125 coefficient matrix walked backward
//!    through every conv in f64 — a ~12-17 GB RSS spike per decision;
//! 2. every candidate child is bounded by
//!    [`BetaCrownVerifier::evaluate_graph_child_bounds`], which runs the FULL
//!    β(+α) optimizer (`beta_iterations` analytical passes + a final pass) at
//!    depth <= `beta_max_depth` — up to `2 * 2k` children x 11 passes per
//!    decision with k = `fsb_candidates`.
//!
//! αβ-CROWN's kFSB and this crate's own wave-batched selector
//! (`multi_objective/batched/kfsb_multi.rs`) do neither: they prescore from the
//! OBJECTIVE's coefficients and bound each candidate child ONCE with the
//! parent's α/β (no optimization). This module is that discipline for the
//! shared graph kFSB selector:
//!
//! * PRESCORE: `compute_graph_babsr_scores_from_bounds` seeded with the
//!   selector's objective row and stopped once every unstable ReLU node is
//!   scored (the objective-directed walk the multi-objective `BoundImpact`
//!   route already uses). `KfsbInterceptOnly` keeps its historical prescore.
//! * SIMULATE: per eval candidate, both children via the same
//!   `GraphBabDomain::with_constraint` the historical scorer calls, each bounded
//!   by ONE `propagate_crown_with_graph_beta` pass over the parent's inherited
//!   node bounds with the child's inherited α and β: the CPU pass of the
//!   non-optimizing arm of `evaluate_graph_child_bounds`, not that whole arm.
//!   The arm first offers the child to the GPU per-domain resnet bound
//!   (`try_gpu_beta_perdomain_bound`) and checks the deadline before its CPU
//!   pass; the single shot does neither (the constrained pass polls the same
//!   effective deadline itself), so where that GPU route is admitted (a sound,
//!   deadline-honouring GPU CROWN backend on a conv resnet) a candidate the
//!   historical scorer bounded on the GPU is a CPU pass here. Advisory either
//!   way: this changes the scorer's cost and numbers, never a published bound
//!   (review(dag-throughput)).
//! * PICK: the shared `reduce_graph_kfsb_scores` / fallback, unchanged.
//!
//! SOUNDNESS: ADVISORY-ONLY. The selector returns only `(node, neuron)` — which
//! ReLU to partition. The two committed children are an exact partition of the
//! parent's region for ANY pick, and they are bounded by the caller's unchanged
//! pipeline; no bound, cache, β, α or verdict computed here is published. A
//! simulation error propagates exactly like the historical serial loop's.
//!
//! #kfsb-production (on review/dag-throughput bd1eabcd9, whose review named
//! these gaps):
//!
//! * MEMORY: the simulations share ONE dense budget
//!   ([`super::kfsb_fanout`]): the live dense budget read once per decision
//!   (one dark pass's claim) — or, under the router's R4 memory lease, a
//!   lease from its envelope — divided across the simulations in flight,
//!   whose number is capped by bytes; the share is installed on every worker
//!   of a pool the fan-out owns, and the candidates run in waves of that
//!   width.
//! * DEVICE: when an engine is present, each candidate child is first offered
//!   to the GPU per-domain β route the historical scorer's non-optimizing arm
//!   uses ([`BetaCrownVerifier::try_gpu_beta_perdomain_bound`]: a sound,
//!   deadline-honouring GPU CROWN on a conv resnet, ReLU splits only); a
//!   refusal falls back to the CPU constrained pass. The device result only
//!   RANKS a candidate (untrusted search): it never certifies anything, and
//!   the observable says which route each simulation took.
//! * OBSERVABLE: the first `KFSB_SINGLE_SHOT_DECISION_LINES` decisions print
//!   their line; each armed lane prints one summary line.

use std::time::Instant;

use ny_core::{GemmEngine, Result};

use crate::beta_crown::branching::{BranchingHeuristic, GraphNeuronConstraint};
use crate::beta_crown::domain::{GraphBabDomain, GraphCrownContext};
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::beta_crown::state::GraphDomainAlphaState;
use crate::GraphNetwork;

use super::super::super::super::BetaCrownVerifier;
use super::super::super::branching::kfsb_shared::{
    select_graph_kfsb_eval_candidates, GraphKfsbCandidate,
};
use super::kfsb_fanout::{
    decision_line, plan_kfsb_fanout, run_shared_budget_fanout, KfsbDecisionRecord, KfsbFanoutPlan,
    KFSB_MIN_SHARE_BYTES,
};

/// Per-decision accounting for the `[kfsb-single-shot]` observable.
#[derive(Debug, Default, Clone, Copy)]
pub(in crate::beta_crown::engine) struct KfsbSingleShotReceipt {
    pub unstable: usize,
    pub scored: usize,
    pub candidates: usize,
    pub record: KfsbDecisionRecord,
}

/// Which route bounded one simulated child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::beta_crown::engine) enum KfsbSimRoute {
    /// The GPU per-domain β route (untrusted search: it only ranks).
    Gpu,
    /// The CPU constrained pass (no engine, or the GPU route refused).
    Cpu,
}

impl BetaCrownVerifier {
    /// #kfsb-single-shot selector body. Called only when
    /// `config.kfsb_single_shot` is set and `fsb_candidates > 0`.
    pub(in crate::beta_crown::engine) fn select_graph_branch_kfsb_single_shot(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
        objective: &[f32],
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(String, usize, f32)> {
        #[cfg(test)]
        KFSB_SINGLE_SHOT_CALLS.with(|calls| calls.set(calls.get() + 1));
        let mut receipt = KfsbSingleShotReceipt {
            unstable: unstable.len(),
            ..KfsbSingleShotReceipt::default()
        };
        let started = Instant::now();
        let scored =
            self.score_graph_kfsb_candidates_objective(graph, domain, unstable, objective)?;
        receipt.record.prescore_secs = started.elapsed().as_secs_f64();
        receipt.scored = scored.len();
        let Some(best_prescore) = scored.first().cloned() else {
            self.emit_receipt(domain.depth, &receipt, None);
            return self.select_graph_branch(graph, domain, unstable);
        };

        let eval_candidates = select_graph_kfsb_eval_candidates(
            &scored,
            self.config.fsb_candidates,
            matches!(self.config.branching_heuristic, BranchingHeuristic::Kfsb),
        );
        receipt.candidates = eval_candidates.len();

        let sim_started = Instant::now();
        let child_bounds = self.collect_graph_kfsb_child_bounds_single_shot(
            graph,
            domain,
            objective,
            engine,
            &eval_candidates,
            &mut receipt.record,
        );
        receipt.record.sim_secs = sim_started.elapsed().as_secs_f64();
        let child_bounds = match child_bounds {
            Ok(child_bounds) => child_bounds,
            Err(error) => {
                self.emit_receipt(domain.depth, &receipt, None);
                return Err(error);
            }
        };

        let best = self.reduce_graph_kfsb_scores(&eval_candidates, &child_bounds);
        if let Some((node_name, neuron_idx, score, _main_score)) = best {
            self.emit_receipt(domain.depth, &receipt, Some((&node_name, neuron_idx)));
            return Ok((node_name, neuron_idx, score));
        }
        let fallback = self.finalize_graph_kfsb_fallback_selection(domain, &scored, best_prescore);
        self.emit_receipt(domain.depth, &receipt, Some((&fallback.0, fallback.1)));
        Ok(fallback)
    }

    /// Record one armed decision and print its line while the lineage is
    /// inside its first `KFSB_SINGLE_SHOT_DECISION_LINES` decisions.
    fn emit_receipt(
        &self,
        depth: usize,
        receipt: &KfsbSingleShotReceipt,
        pick: Option<(&str, usize)>,
    ) {
        if self
            .kfsb_single_shot_telemetry
            .record_decision(&receipt.record)
        {
            eprintln!(
                "{}",
                decision_line(
                    depth,
                    receipt.unstable,
                    receipt.scored,
                    receipt.candidates,
                    &receipt.record,
                    pick,
                )
            );
        }
    }

    /// Prescore every unstable neuron. `Kfsb` / `FilteredSmartBranching` seed
    /// the BaBSR walk with `objective` (one row) and stop it once every
    /// unstable ReLU node has been scored; `KfsbInterceptOnly` keeps the
    /// historical intercept prescore. Sorted by main score, NaN last — the same
    /// order [`Self::score_graph_kfsb_candidates`] produces.
    fn score_graph_kfsb_candidates_objective(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
        objective: &[f32],
    ) -> Result<Vec<GraphKfsbCandidate>> {
        let reduce_op = match self.config.branching_heuristic {
            BranchingHeuristic::Kfsb => Some(self.config.kfsb_reduce_op),
            BranchingHeuristic::FilteredSmartBranching => {
                Some(crate::beta_crown::config::KfsbReduceOp::Min)
            }
            _ => None,
        };
        let Some(reduce_op) = reduce_op else {
            return self.score_graph_kfsb_candidates(graph, domain, unstable);
        };
        let unstable_nodes: std::collections::HashSet<String> =
            unstable.iter().map(|(name, _)| name.clone()).collect();
        let babsr_scores = self.compute_graph_babsr_scores_from_bounds(
            graph,
            &domain.node_bounds,
            &domain.input_bounds,
            reduce_op,
            Some(objective),
            Some(&unstable_nodes),
        )?;
        let mut scored = Vec::with_capacity(unstable.len());
        for (node_name, neuron_idx) in unstable {
            let parts = babsr_scores
                .get(&(node_name.clone(), *neuron_idx))
                .copied()
                .unwrap_or_default();
            scored.push(GraphKfsbCandidate {
                node_name: node_name.clone(),
                neuron_idx: *neuron_idx,
                main_score: parts.main_score,
                backup_score: parts.backup_score,
            });
        }
        scored.sort_by(|a, b| {
            crate::cmp_utils::nan_last_descending_cmp(&a.main_score, &b.main_score)
        });
        Ok(scored)
    }

    /// One constrained CROWN pass per candidate child, with the parent's
    /// inherited node bounds and the child's inherited α/β. The `(active,
    /// inactive)` mapping is exactly the historical serial loop's: an
    /// infeasible child or non-finite bounds is `None`; an error propagates.
    ///
    /// The candidates are simulated in PARALLEL under ONE shared dense budget
    /// ([`super::kfsb_fanout`]): the live budget read once here (or the
    /// router's R4 lease), divided across the simulations in flight, whose
    /// number is capped by bytes; each worker of the fan-out's own pool
    /// carries the share. Each simulation reads only shared immutable state
    /// (graph, parent domain, objective, the verifier's `&self`, a `Sync`
    /// engine) and builds its own child, so the per-candidate results are
    /// independent of scheduling; they are collected in candidate order and
    /// the FIRST error in candidate order is returned, so the pick and the
    /// error disposition are the serial loop's. The Complete Clip suppression
    /// is a verifier-wide token multiset (not thread-local), so it covers
    /// every worker.
    fn collect_graph_kfsb_child_bounds_single_shot(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        objective: &[f32],
        engine: Option<&dyn GemmEngine>,
        eval_candidates: &[GraphKfsbCandidate],
        record: &mut KfsbDecisionRecord,
    ) -> Result<Vec<(Option<(f32, f32)>, Option<(f32, f32)>)>> {
        let authority = ny_core::memory_lease::global();
        let lease = self.kfsb_single_shot_telemetry.fanout_lease(authority);
        let threads = rayon::current_num_threads();
        let plan = plan_kfsb_fanout(
            crate::network::crown_memory::cpu_crown_dense_budget_bytes(),
            lease.decision,
            eval_candidates.len(),
            threads,
            KFSB_MIN_SHARE_BYTES,
        );
        let result = self.collect_graph_kfsb_child_bounds_planned(
            graph,
            domain,
            objective,
            engine,
            eval_candidates,
            &plan,
            threads,
            record,
        );
        self.kfsb_single_shot_telemetry
            .finish_lease(authority, &lease);
        result
    }

    /// The simulations of one decision under an explicit fan-out `plan`.
    #[allow(clippy::too_many_arguments)]
    fn collect_graph_kfsb_child_bounds_planned(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        objective: &[f32],
        engine: Option<&dyn GemmEngine>,
        eval_candidates: &[GraphKfsbCandidate],
        plan: &KfsbFanoutPlan,
        threads: usize,
        record: &mut KfsbDecisionRecord,
    ) -> Result<Vec<(Option<(f32, f32)>, Option<(f32, f32)>)>> {
        // Candidate-side propagation is advisory: as in the historical
        // `estimate_graph_kfsb_child_bounds`, do not spend the verdict-path
        // Complete Clip budget on discarded children.
        let _complete_clip_suppression = self
            .complete_clip_deadline_overrides
            .suppress_complete_clip_scoped();
        record.plan = Some(*plan);
        // The device route is offered only to an engine that exposes a SOUND
        // GPU CROWN backend (the route's own first filter); a CPU engine is
        // not a refusal.
        let gpu_capable = engine
            .and_then(|engine| engine.as_gpu_crown_backward())
            .is_some_and(|gpu| gpu.provides_sound_gpu_crown());
        let pool_started = Instant::now();
        let first_sim = std::sync::atomic::AtomicU64::new(u64::MAX);
        let per_candidate: Vec<Result<SimulatedCandidate>> =
            run_shared_budget_fanout(eval_candidates, plan, threads, |candidate| {
                let offset = u64::try_from(pool_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                first_sim.fetch_min(offset, std::sync::atomic::Ordering::Relaxed);
                let simulated = self.simulate_kfsb_candidate(
                    graph,
                    domain,
                    objective,
                    engine,
                    gpu_capable,
                    candidate,
                );
                // Test probe: the dense budget this simulation's worker sees
                // (the share it ran under), read on that worker.
                #[cfg(test)]
                let simulated = simulated.map(|mut done| {
                    done.budget_seen = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
                    done
                });
                simulated
            });
        let first = first_sim.load(std::sync::atomic::Ordering::Relaxed);
        if first != u64::MAX {
            record.pool_secs = first as f64 / 1e9;
        }
        let mut child_bounds = Vec::with_capacity(eval_candidates.len());
        for simulated in per_candidate {
            let simulated = simulated?;
            record.sims += simulated.sims;
            record.infeasible += simulated.infeasible;
            record.gpu_sims += simulated.gpu;
            record.cpu_sims += simulated.cpu;
            record.gpu_refused += simulated.gpu_refused;
            #[cfg(test)]
            {
                record.max_budget_seen = record.max_budget_seen.max(simulated.budget_seen);
            }
            child_bounds.push((simulated.sides[0], simulated.sides[1]));
        }
        Ok(child_bounds)
    }

    /// Both children of one candidate, each bounded by ONE pass: the GPU
    /// per-domain β route when the engine has a sound GPU CROWN backend and
    /// the route admits the child, else the CPU constrained pass.
    fn simulate_kfsb_candidate(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        objective: &[f32],
        engine: Option<&dyn GemmEngine>,
        gpu_capable: bool,
        candidate: &GraphKfsbCandidate,
    ) -> Result<SimulatedCandidate> {
        let mut simulated = SimulatedCandidate::default();
        for (slot, is_active) in [true, false].into_iter().enumerate() {
            let constraint = GraphNeuronConstraint {
                node_name: candidate.node_name.clone(),
                neuron_idx: candidate.neuron_idx,
                is_active,
                score: candidate.main_score,
            };
            let Some(mut child) =
                domain.with_constraint(graph, constraint, self.config.verify_upper_bound)?
            else {
                simulated.infeasible += 1;
                continue;
            };
            if child.alpha_state.is_empty() {
                child.alpha_state = GraphDomainAlphaState::from_graph_bounds(
                    graph,
                    &child.node_bounds,
                    &child.history,
                    child.input_bounds.as_ref(),
                );
            }
            simulated.sims += 1;
            let (bounds, route) =
                self.simulate_kfsb_child(graph, domain, &child, objective, engine, gpu_capable)?;
            match route {
                KfsbSimRoute::Gpu => simulated.gpu += 1,
                KfsbSimRoute::Cpu => simulated.cpu += 1,
            }
            if gpu_capable && route == KfsbSimRoute::Cpu {
                simulated.gpu_refused += 1;
            }
            let (lower, upper) = bounds;
            if lower.is_finite() && upper.is_finite() {
                simulated.sides[slot] = Some((lower, upper));
            }
        }
        Ok(simulated)
    }

    /// ONE pass over one simulated child: the GPU per-domain β route first
    /// when the engine has a sound GPU CROWN backend (UNTRUSTED for this
    /// purpose: its number only ranks the candidate; a refusal is `None`),
    /// else — or on refusal — the CPU constrained pass with the child's
    /// inherited α and β over the parent's node bounds.
    fn simulate_kfsb_child(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        child: &GraphBabDomain,
        objective: &[f32],
        engine: Option<&dyn GemmEngine>,
        gpu_capable: bool,
    ) -> Result<((f32, f32), KfsbSimRoute)> {
        if gpu_capable {
            if let Some((lower, upper, _node_bounds)) = self.try_gpu_beta_perdomain_bound(
                graph,
                child,
                &domain.node_bounds,
                objective,
                engine,
            ) {
                return Ok(((lower, upper), KfsbSimRoute::Gpu));
            }
        }
        let mut context =
            GraphCrownContext::new(&child.history, None, Some(&domain.node_bounds), engine)
                .with_delta_seeds(&child.delta_pre_nodes);
        if !child.alpha_state.is_empty() {
            context = context.with_alpha(&child.alpha_state);
        }
        let (output, _node_cache) = self.propagate_crown_with_graph_beta(
            graph,
            child.input_bounds.as_ref(),
            &context,
            &child.beta_state,
            Some(objective),
        )?;
        Ok((
            (output.lower_scalar(), output.upper_scalar()),
            KfsbSimRoute::Cpu,
        ))
    }
}

/// One candidate's simulated `(active, inactive)` child bounds plus its
/// receipt counters.
#[derive(Debug, Default, Clone, Copy)]
struct SimulatedCandidate {
    sides: [Option<(f32, f32)>; 2],
    sims: usize,
    infeasible: usize,
    gpu: usize,
    cpu: usize,
    gpu_refused: usize,
    /// Test probe: the dense budget the simulation's worker saw.
    #[cfg(test)]
    budget_seen: usize,
}

/// Test seam: one single-shot simulation pass over `candidates`, exactly as
/// the armed selector runs it (CPU, no engine).
#[cfg(test)]
impl BetaCrownVerifier {
    pub(in crate::beta_crown::engine) fn kfsb_single_shot_child_bounds_for_tests(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        objective: &[f32],
        candidates: &[GraphKfsbCandidate],
    ) -> Result<Vec<(Option<(f32, f32)>, Option<(f32, f32)>)>> {
        let mut record = KfsbDecisionRecord::default();
        self.collect_graph_kfsb_child_bounds_single_shot(
            graph,
            domain,
            objective,
            None,
            candidates,
            &mut record,
        )
    }

    /// Test seam: one simulation pass under an explicit fan-out `plan` and
    /// engine, returning the bounds and the decision's record (routes,
    /// sims, the plan it ran under).
    pub(in crate::beta_crown::engine) fn kfsb_single_shot_child_bounds_planned_for_tests(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        objective: &[f32],
        engine: Option<&dyn GemmEngine>,
        candidates: &[GraphKfsbCandidate],
        plan: &KfsbFanoutPlan,
        threads: usize,
    ) -> Result<(
        Vec<(Option<(f32, f32)>, Option<(f32, f32)>)>,
        KfsbDecisionRecord,
    )> {
        let mut record = KfsbDecisionRecord::default();
        let bounds = self.collect_graph_kfsb_child_bounds_planned(
            graph,
            domain,
            objective,
            engine,
            candidates,
            plan,
            threads,
            &mut record,
        )?;
        Ok((bounds, record))
    }

    /// Test seam: the armed PRESCORE ranking (main score descending, NaN
    /// last), exactly as the armed selector computes it before candidate
    /// filtering (review(dag-throughput): the end-to-end pick cannot see it
    /// when every unstable neuron is an eval candidate).
    pub(in crate::beta_crown::engine) fn kfsb_single_shot_prescore_for_tests(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
        objective: &[f32],
    ) -> Result<Vec<GraphKfsbCandidate>> {
        self.score_graph_kfsb_candidates_objective(graph, domain, unstable, objective)
    }
}

#[cfg(test)]
thread_local! {
    static KFSB_SINGLE_SHOT_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Number of armed single-shot DECISIONS made on this thread (test-only; the
/// multi-objective sequential lane selects on the caller's thread).
#[cfg(test)]
pub(in crate::beta_crown::engine) fn kfsb_single_shot_calls_for_tests() -> u64 {
    KFSB_SINGLE_SHOT_CALLS.with(std::cell::Cell::get)
}
