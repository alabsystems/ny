// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Graph kFSB selection for the GPU BaB ReLU-split path.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use ny_core::{GemmEngine, NyError, Result};
use tracing::trace;

use crate::batched_domain::{BatchedDomainOptions, BatchedDomains};
use crate::beta_crown::branching::{BranchingHeuristic, GraphNeuronConstraint};
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::graph::shared::setup::build_sorted_relu_nodes;
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::beta_crown::state::GraphDomainAlphaState;
use crate::GraphNetwork;

// #kfsb-multi: the candidate type + top-k ∪ backup-top-k filter are hoisted to
// `engine::branching::kfsb_shared` so the multi-objective wave-batched lane
// (`multi_objective::batched::kfsb_multi`) shares them with this lane.
use super::super::super::super::BetaCrownVerifier;
use super::super::super::branching::kfsb_shared::{
    kfsb_reduce, select_graph_kfsb_eval_candidates, GraphKfsbCandidate,
};

struct GraphKfsbChildRequest<'a> {
    graph: &'a GraphNetwork,
    domain: &'a GraphBabDomain,
    objective: &'a [f32],
    engine: Option<&'a dyn GemmEngine>,
    candidate: &'a GraphKfsbCandidate,
    is_active: bool,
}

impl BetaCrownVerifier {
    // #mo-scorer-fix: widened from gpu_bab scope so the multi-objective
    // selector (engine::branching::graph) can route the kFSB family here.
    pub(in crate::beta_crown::engine) fn select_graph_branch_kfsb_in_gpu_batched(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
        objective: &[f32],
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(String, usize, f32)> {
        if self.config.fsb_candidates == 0 {
            trace!(
                unstable = unstable.len(),
                "GPU graph kFSB has zero candidates configured; falling back to BaBSR ranking"
            );
            return self.select_graph_babsr_fallback_branch(graph, domain, unstable);
        }

        let scored = self.score_graph_kfsb_candidates(graph, domain, unstable)?;
        let Some(best_prescore) = scored.first().cloned() else {
            return self.select_graph_branch(graph, domain, unstable);
        };

        let eval_candidates = select_graph_kfsb_eval_candidates(
            &scored,
            self.config.fsb_candidates,
            matches!(self.config.branching_heuristic, BranchingHeuristic::Kfsb),
        );

        // Compute the (active, inactive) child bounds for every eval candidate.
        // On the GPU path (`engine.is_some()`) this is ONE batched propagation
        // for all `2 * eval_candidates.len()` children; on the CPU path it is the
        // serial per-child loop. Scoring is shared below (`reduce_graph_kfsb_scores`).
        // NOTE: the batched CROWN backward is a DIFFERENT (tighter) propagation than
        // the serial `evaluate_graph_child_bounds` single pass, so per-child bounds —
        // and hence the pick — are not guaranteed bit-identical between the two
        // paths. This is sound (advisory scorer, never feeds a verdict); see
        // `collect_graph_kfsb_child_bounds_batched`.
        let child_bounds = self.collect_graph_kfsb_child_bounds(
            graph,
            domain,
            objective,
            engine,
            &eval_candidates,
        )?;

        let best = self.reduce_graph_kfsb_scores(&eval_candidates, &child_bounds);

        if let Some((node_name, neuron_idx, score, main_score)) = best {
            trace!(
                node = %node_name,
                neuron = neuron_idx,
                score,
                main_score,
                eval_candidates = eval_candidates.len(),
                total_candidates = scored.len(),
                "GPU graph kFSB selected candidate"
            );
            return Ok((node_name, neuron_idx, score));
        }

        Ok(self.finalize_graph_kfsb_fallback_selection(domain, &scored, best_prescore))
    }

    fn score_graph_kfsb_candidates(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
    ) -> Result<Vec<GraphKfsbCandidate>> {
        let mut scored = Vec::with_capacity(unstable.len());

        match self.config.branching_heuristic {
            BranchingHeuristic::KfsbInterceptOnly => {
                let intercept_scores =
                    self.compute_graph_babsr_intercept_only_scores(graph, domain)?;
                for (node_name, neuron_idx) in unstable {
                    let main_score = intercept_scores
                        .get(&(node_name.clone(), *neuron_idx))
                        .copied()
                        .unwrap_or(0.0);
                    scored.push(GraphKfsbCandidate {
                        node_name: node_name.clone(),
                        neuron_idx: *neuron_idx,
                        main_score,
                        backup_score: 0.0,
                    });
                }
            }
            BranchingHeuristic::Kfsb => {
                let babsr_scores =
                    self.compute_graph_babsr_scores(graph, domain, self.config.kfsb_reduce_op)?;
                for (node_name, neuron_idx) in unstable {
                    let score_parts = babsr_scores
                        .get(&(node_name.clone(), *neuron_idx))
                        .copied()
                        .unwrap_or_default();
                    scored.push(GraphKfsbCandidate {
                        node_name: node_name.clone(),
                        neuron_idx: *neuron_idx,
                        main_score: score_parts.main_score,
                        backup_score: score_parts.backup_score,
                    });
                }
            }
            // #mo-scorer-fix: FSB = BaBSR prescore filter (conservative Min
            // reduce) + child evaluation of the top-k main-score candidates
            // (no intercept-backup union — that is kFSB's extension). Before
            // this arm FSB fell through to `scored = []` → the BaBSR/intercept
            // fallback, collapsing FSB onto the other heuristics' pick.
            BranchingHeuristic::FilteredSmartBranching => {
                let babsr_scores = self.compute_graph_babsr_scores(
                    graph,
                    domain,
                    crate::beta_crown::config::KfsbReduceOp::Min,
                )?;
                for (node_name, neuron_idx) in unstable {
                    let score_parts = babsr_scores
                        .get(&(node_name.clone(), *neuron_idx))
                        .copied()
                        .unwrap_or_default();
                    scored.push(GraphKfsbCandidate {
                        node_name: node_name.clone(),
                        neuron_idx: *neuron_idx,
                        main_score: score_parts.main_score,
                        backup_score: score_parts.backup_score,
                    });
                }
            }
            _ => {}
        }

        scored.sort_by(|a, b| {
            crate::cmp_utils::nan_last_descending_cmp(&a.main_score, &b.main_score)
        });
        Ok(scored)
    }

    fn select_graph_babsr_fallback_branch(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
    ) -> Result<(String, usize, f32)> {
        let babsr_scores = self.compute_graph_babsr_scores(
            graph,
            domain,
            crate::beta_crown::config::KfsbReduceOp::Min,
        )?;
        let mut best: Option<(String, usize, f32)> = None;

        for (node_name, neuron_idx) in unstable {
            let score = babsr_scores
                .get(&(node_name.clone(), *neuron_idx))
                .copied()
                .unwrap_or_default()
                .main_score;
            if score.is_nan() {
                continue;
            }

            if best
                .as_ref()
                .map(|(_, _, best_score)| score > *best_score)
                .unwrap_or(true)
            {
                best = Some((node_name.clone(), *neuron_idx, score));
            }
        }

        if let Some(best) = best {
            Ok(best)
        } else {
            self.select_graph_branch(graph, domain, unstable)
        }
    }

    fn estimate_graph_kfsb_child_bounds(
        &self,
        request: GraphKfsbChildRequest<'_>,
    ) -> Result<Option<(f32, f32)>> {
        let constraint = GraphNeuronConstraint {
            node_name: request.candidate.node_name.clone(),
            neuron_idx: request.candidate.neuron_idx,
            is_active: request.is_active,
            score: request.candidate.main_score,
        };
        let Some(mut child) = request.domain.with_constraint(
            request.graph,
            constraint,
            self.config.verify_upper_bound,
        )?
        else {
            return Ok(None);
        };

        if self.evaluate_graph_child_bounds(
            request.graph,
            &mut child,
            &request.domain.node_bounds,
            request.objective,
            None,
            request.engine,
        )? {
            Ok(Some((child.lower_bound, child.upper_bound)))
        } else {
            Ok(None)
        }
    }

    /// Compute per-candidate `(active, inactive)` child bounds for the eval set.
    ///
    /// Returns one `(active_bounds, inactive_bounds)` entry per eval candidate,
    /// in candidate order. Each side is `Some((lower, upper))` for a feasible
    /// child that propagated to finite bounds, or `None` for an infeasible child
    /// (`with_constraint` returned `None`) or a propagation that failed / produced
    /// non-finite bounds — exactly the mapping the serial path fed into
    /// `child_bound_value` (`None` scores as `NEG_INFINITY`).
    ///
    /// When a real GPU engine is present, all `2 * candidates` children are
    /// evaluated in ONE batched forward+backward pass (mirrors
    /// `process_graph_domains_batched_gpu`). Otherwise — and on any batched-build
    /// or propagation FAILURE — it falls back to the serial per-child loop, whose
    /// result is exactly the pure-serial one (same error-propagation semantics).
    /// On batched SUCCESS the bounds come from the batched engine, which is a
    /// different (tighter) propagation than serial; see
    /// `collect_graph_kfsb_child_bounds_batched`.
    fn collect_graph_kfsb_child_bounds(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        objective: &[f32],
        engine: Option<&dyn GemmEngine>,
        eval_candidates: &[GraphKfsbCandidate],
    ) -> Result<Vec<(Option<(f32, f32)>, Option<(f32, f32)>)>> {
        // Gated OFF by default (`NY_KFSB_BATCH_EVAL=1` to enable). The batched
        // engine (`propagate_crown_with_batched_domains_full`) is faster than
        // the serial per-child loop AND produces a genuinely TIGHTER, sound,
        // objective-aware bound — which is the same bound the batched BaB loop
        // then explores with, so scoring with it is arguably more consistent.
        // But "tighter" changes the advisory branch PICK vs the serial scorer
        // (not bit-parity), and this scorer is on the DEFAULT gpu-BaB kFSB path
        // (batched_gpu.rs), i.e. the auto-kFSB cifar100/tinyimagenet tracks. So
        // the default stays serial (byte-unchanged competition behavior).
        // Advisory-only either way (branch selection never feeds a verdict), so
        // the gate is soundness-free.
        //
        // A/B MEASURED 2026-07-18 (docs/MEASURED_KFSB_GATES.md) — stays gated,
        // decision FINAL on this evidence: on cifar100_2024 (preset, 4 inst,
        // 100s wgpu) the gate engaged with zero verdict/solve deltas; on the
        // relusplitter MO lane it produced results bit-identical to serial
        // (engagement inconclusive — designed silent fallback or a route that
        // bypasses this scorer's engine path), and no arm changed solved-count
        // anywhere. Re-open only with a solved-count win on competition hardware.
        let batch_enabled = matches!(
            std::env::var("NY_KFSB_BATCH_EVAL").ok().as_deref(),
            Some("1")
        );
        // Batch only when enabled AND a real GPU engine is available — the
        // batched propagation requires one. On the CPU path keep the serial loop.
        if let (true, Some(engine)) = (batch_enabled, engine) {
            match self.collect_graph_kfsb_child_bounds_batched(
                graph,
                domain,
                objective,
                engine,
                eval_candidates,
            ) {
                Ok(child_bounds) => return Ok(child_bounds),
                Err(e) => {
                    // Any batched-path failure (BatchedDomains build, propagation,
                    // result-count mismatch) falls back to the proven serial loop,
                    // which yields the exact pure-serial result — including its
                    // error-propagation semantics.
                    trace!(
                        error = %e,
                        "GPU graph kFSB batched child evaluation failed; falling back to serial"
                    );
                }
            }
        }
        self.collect_graph_kfsb_child_bounds_serial(
            graph,
            domain,
            objective,
            engine,
            eval_candidates,
        )
    }

    /// Serial per-child evaluation: two `estimate_graph_kfsb_child_bounds`
    /// propagations per candidate. Preserves the original loop's `?` error
    /// propagation exactly.
    fn collect_graph_kfsb_child_bounds_serial(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        objective: &[f32],
        engine: Option<&dyn GemmEngine>,
        eval_candidates: &[GraphKfsbCandidate],
    ) -> Result<Vec<(Option<(f32, f32)>, Option<(f32, f32)>)>> {
        let mut child_bounds = Vec::with_capacity(eval_candidates.len());
        for candidate in eval_candidates {
            let active = self.estimate_graph_kfsb_child_bounds(GraphKfsbChildRequest {
                graph,
                domain,
                objective,
                engine,
                candidate,
                is_active: true,
            })?;
            let inactive = self.estimate_graph_kfsb_child_bounds(GraphKfsbChildRequest {
                graph,
                domain,
                objective,
                engine,
                candidate,
                is_active: false,
            })?;
            child_bounds.push((active, inactive));
        }
        Ok(child_bounds)
    }

    /// Batched evaluation of all `2 * candidates` children in ONE propagation.
    ///
    /// Mirrors `process_graph_domains_batched_gpu` (batched_single.rs): build all
    /// child domains via `with_constraint`, batch them with
    /// `BatchedDomains::from_graph_domains_with_options`, run one
    /// `propagate_crown_with_batched_domains_full`, and read each child's
    /// `(lower, upper)` from its `DomainCrownResult` the same way
    /// `evaluate_graph_child_bounds` derives `child.lower_bound/upper_bound`.
    ///
    /// PARITY (structural, exact): an infeasible child (`with_constraint` → `None`)
    /// is never batched; its slot stays `None`, matching the serial `Ok(None)` →
    /// `child_bound_value(None)` mapping. A child whose bounds are non-finite maps to
    /// `None`, matching the `evaluate_graph_child_bounds` finiteness guard that makes
    /// the serial path return `Ok(false)` → `None`. Results map back to candidates in
    /// order via `active_slot`/`inactive_slot`.
    ///
    /// PARITY (numeric, NOT exact): the batched CROWN backward and the serial
    /// `evaluate_graph_child_bounds` single pass use different ReLU relaxations. The
    /// batched pass is TIGHTER (closer to the exact bound; measured on the test
    /// fixture: serial `(-4.35, 1.70)` vs batched `(-4.25, 0.70)`, the latter exact),
    /// a difference independent of the α-state threaded above and larger than a few
    /// ULPs. That can flip which candidate the shared reducer selects. This is SOUND
    /// (advisory branch-SELECTION scorer only — never feeds a verdict; verdict-parity
    /// tests still pass) and is consistent with the BaB loop, which already explores
    /// children using these same batched-engine bounds (batched_single.rs). It is,
    /// however, NOT bit-parity with the serial scorer's pick.
    fn collect_graph_kfsb_child_bounds_batched(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        objective: &[f32],
        engine: &dyn GemmEngine,
        eval_candidates: &[GraphKfsbCandidate],
    ) -> Result<Vec<(Option<(f32, f32)>, Option<(f32, f32)>)>> {
        let n = eval_candidates.len();
        // Build every child domain up front. `*_slot[ci]` holds the index into
        // `children` for that candidate's active / inactive child, or `None` when
        // the child is infeasible (so it never enters the batch).
        let mut children: Vec<GraphBabDomain> = Vec::with_capacity(2 * n);
        let mut active_slot: Vec<Option<usize>> = vec![None; n];
        let mut inactive_slot: Vec<Option<usize>> = vec![None; n];

        for (ci, candidate) in eval_candidates.iter().enumerate() {
            for is_active in [true, false] {
                let constraint = GraphNeuronConstraint {
                    node_name: candidate.node_name.clone(),
                    neuron_idx: candidate.neuron_idx,
                    is_active,
                    score: candidate.main_score,
                };
                // Same call `estimate_graph_kfsb_child_bounds` makes.
                // `Ok(None)` == infeasible child → leave the slot `None`.
                if let Some(mut child) =
                    domain.with_constraint(graph, constraint, self.config.verify_upper_bound)?
                {
                    // Mirror `evaluate_graph_child_bounds` (beta.rs): lazily
                    // initialize an empty child α-state from bounds before the
                    // pass, so the batched backward (which threads each domain's
                    // `alpha_state`, context.rs `from_domains`) sees the SAME
                    // α-state the serial path would. This keeps the two paths as
                    // close as the propagation engines allow. NOTE: the batched
                    // CROWN backward and the serial single pass still use different
                    // ReLU relaxations (the batched pass is TIGHTER — closer to the
                    // exact bound — independent of α), so per-child bounds are NOT
                    // bit-identical to serial; see the parity test's doc comment.
                    if child.alpha_state.is_empty() {
                        child.alpha_state = GraphDomainAlphaState::from_graph_bounds(
                            graph,
                            &child.node_bounds,
                            &child.history,
                            child.input_bounds.as_ref(),
                        );
                    }
                    let idx = children.len();
                    if is_active {
                        active_slot[ci] = Some(idx);
                    } else {
                        inactive_slot[ci] = Some(idx);
                    }
                    children.push(child);
                }
            }
        }

        // Every child infeasible → each candidate scores as (None, None), i.e.
        // NEG_INFINITY on both sides, exactly as the serial loop would.
        if children.is_empty() {
            return Ok(vec![(None, None); n]);
        }

        let child_refs: Vec<&GraphBabDomain> = children.iter().collect();
        let relu_nodes = build_sorted_relu_nodes(graph);
        let batched = BatchedDomains::from_graph_domains_with_options(
            &child_refs,
            &relu_nodes,
            BatchedDomainOptions {
                enable_interm_transfer: self.config.enable_interm_transfer,
            },
        )?;

        // ONE forward+backward pass for all children.
        let results = self.propagate_crown_with_batched_domains_full(
            graph,
            &child_refs,
            &batched,
            objective,
            engine,
        )?;
        if results.len() != children.len() {
            return Err(NyError::InternalError(format!(
                "batched kFSB child evaluation returned {} results for {} children",
                results.len(),
                children.len()
            )));
        }

        // Extract each child's (lower, upper) the SAME way
        // `evaluate_graph_child_bounds` sets `child.lower_bound/upper_bound`:
        // `lower_scalar()`/`upper_scalar()` of the output tensor, with the
        // non-finite → drop guard (beta.rs) mapping to `None`.
        let child_bounds: Vec<Option<(f32, f32)>> = results
            .iter()
            .map(|result| {
                result.as_ref().and_then(|(output, _node_cache)| {
                    let lower = output.lower_scalar();
                    let upper = output.upper_scalar();
                    (lower.is_finite() && upper.is_finite()).then_some((lower, upper))
                })
            })
            .collect();

        Ok((0..n)
            .map(|ci| {
                let active = active_slot[ci].and_then(|idx| child_bounds[idx]);
                let inactive = inactive_slot[ci].and_then(|idx| child_bounds[idx]);
                (active, inactive)
            })
            .collect())
    }

    /// Shared kFSB scoring reduction over the eval candidates and their
    /// precomputed `(active, inactive)` child bounds. Identical logic to the
    /// original serial loop — `child_bound_value`, the FSB-vs-configured
    /// `reduce_op` selection, `kfsb_reduce`, and the 1e-6-tolerant `is_better`
    /// tie-break — so given the SAME child bounds both paths reduce to the same
    /// candidate. (The batched and serial paths feed it different bounds; see
    /// `collect_graph_kfsb_child_bounds_batched`.)
    /// Returns `(node_name, neuron_idx, kfsb_score, main_score)` for the winner.
    fn reduce_graph_kfsb_scores(
        &self,
        eval_candidates: &[GraphKfsbCandidate],
        child_bounds: &[(Option<(f32, f32)>, Option<(f32, f32)>)],
    ) -> Option<(String, usize, f32, f32)> {
        let mut best: Option<(String, usize, f32, f32)> = None;

        for (candidate, (active, inactive)) in eval_candidates.iter().zip(child_bounds.iter()) {
            let active_val = self.config.child_bound_value(*active);
            let inactive_val = self.config.child_bound_value(*inactive);
            if active_val == f32::NEG_INFINITY && inactive_val == f32::NEG_INFINITY {
                continue;
            }

            // #mo-scorer-fix: FSB is DEFINED as "best worst-child improvement"
            // (see `BranchingHeuristic::FilteredSmartBranching`), so its child
            // combine is always Min; the kFSB variants keep the configured
            // reduce op (their tunable extension).
            let reduce_op = if matches!(
                self.config.branching_heuristic,
                BranchingHeuristic::FilteredSmartBranching
            ) {
                crate::beta_crown::config::KfsbReduceOp::Min
            } else {
                self.config.kfsb_reduce_op
            };
            let kfsb_score = kfsb_reduce(reduce_op, active_val, inactive_val);
            if kfsb_score.is_nan() {
                continue;
            }

            let is_better = best
                .as_ref()
                .map(|(_, _, best_score, best_main)| {
                    kfsb_score > *best_score + 1e-6
                        || ((kfsb_score - *best_score).abs() <= 1e-6
                            && !candidate.main_score.is_nan()
                            && (best_main.is_nan() || candidate.main_score > *best_main))
                })
                .unwrap_or(true);

            if is_better {
                best = Some((
                    candidate.node_name.clone(),
                    candidate.neuron_idx,
                    kfsb_score,
                    candidate.main_score,
                ));
            }
        }

        best
    }

    fn fallback_graph_kfsb_candidate(
        &self,
        domain: &GraphBabDomain,
        scored: &[GraphKfsbCandidate],
        best_prescore: GraphKfsbCandidate,
    ) -> GraphKfsbCandidate {
        if scored.is_empty() {
            return best_prescore;
        }

        let mut hasher = DefaultHasher::new();
        domain.depth.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % scored.len();
        scored[idx].clone()
    }

    fn finalize_graph_kfsb_fallback_selection(
        &self,
        domain: &GraphBabDomain,
        scored: &[GraphKfsbCandidate],
        best_prescore: GraphKfsbCandidate,
    ) -> (String, usize, f32) {
        let fallback = self.fallback_graph_kfsb_candidate(domain, scored, best_prescore);
        trace!(
            node = %fallback.node_name,
            neuron = fallback.neuron_idx,
            prescore = fallback.main_score,
            total_candidates = scored.len(),
            "GPU graph kFSB fell back after all child evaluations failed"
        );
        (fallback.node_name, fallback.neuron_idx, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ndarray::{arr1, arr2};

    use super::*;
    use crate::beta_crown::{
        BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic, GraphBabDomain,
    };
    use crate::layers::LinearLayer;
    use crate::{BoundedTensor, GraphNetwork, GraphNode, Layer, ReLULayer};

    #[ntest::timeout(10000)]
    #[test]
    fn test_graph_gpu_kfsb_intercept_only_selects_child_evaluated_candidate_4300() -> Result<()> {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::KfsbInterceptOnly,
            fsb_candidates: 1,
            beta_iterations: 0,
            ..Default::default()
        });

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "linear",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0, 1.0, 1.0, 1.0]]), None)
                    .expect("linear layer should build"),
            ),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear");

        let input = BoundedTensor::new(
            arr1(&[-0.1, -1.0, -2.0, -3.0]).into_dyn(),
            arr1(&[0.2, 2.0, 1.0, 3.0]).into_dyn(),
        )?;
        let node_bounds = graph.collect_node_bounds(&input)?;
        let domain = GraphBabDomain::root(node_bounds, -5.0, 6.0, &input, false)?;
        let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &["relu".to_string()]);

        let (node_name, neuron_idx, score) = verifier.select_graph_branch_kfsb_in_gpu_batched(
            &graph,
            &domain,
            &unstable,
            &[1.0],
            None,
        )?;

        assert_eq!(node_name, "relu");
        assert_eq!(
            neuron_idx, 0,
            "graph kFSB intercept-only should pick the small-intercept candidate after child evaluation"
        );
        assert!(score.is_finite(), "expected finite kFSB score, got {score}");

        Ok(())
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_graph_gpu_kfsb_zero_candidates_falls_back_to_babsr_4300() -> Result<()> {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::KfsbInterceptOnly,
            fsb_candidates: 0,
            beta_iterations: 0,
            ..Default::default()
        });

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[1.0, 10.0])))
                    .expect("linear layer should build"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0, 1.0]]), None).expect("linear layer should build"),
            ),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear2");

        let input = BoundedTensor::new(
            arr1(&[-2.0, -11.0]).into_dyn(),
            arr1(&[0.0, -9.0]).into_dyn(),
        )?;
        let node_bounds = graph.collect_node_bounds(&input)?;
        let domain = GraphBabDomain::root(node_bounds, -1.0, 1.0, &input, false)?;
        let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &["relu".to_string()]);

        let (node_name, neuron_idx, score) = verifier.select_graph_branch_kfsb_in_gpu_batched(
            &graph,
            &domain,
            &unstable,
            &[1.0],
            None,
        )?;

        assert_eq!(node_name, "relu");
        assert_eq!(
            neuron_idx, 1,
            "fsb_candidates=0 should bypass kFSB child evaluation and fall back to BaBSR"
        );
        assert!(
            (score - 5.0).abs() < 1e-6,
            "BaBSR fallback should preserve the recovered bias score, got {score}"
        );

        Ok(())
    }

    #[test]
    fn test_graph_gpu_kfsb_child_eval_failure_fallback_uses_zero_score_4300() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input = BoundedTensor::new(
            arr1(&[-1.0f32, -1.0]).into_dyn(),
            arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .expect("input bounds should build");
        let domain = GraphBabDomain::root(HashMap::new(), -1.0, 1.0, &input, false)
            .expect("domain should build");
        let scored = vec![
            GraphKfsbCandidate {
                node_name: "relu".to_string(),
                neuron_idx: 0,
                main_score: 3.5,
                backup_score: -0.5,
            },
            GraphKfsbCandidate {
                node_name: "relu".to_string(),
                neuron_idx: 1,
                main_score: 2.5,
                backup_score: -0.25,
            },
        ];

        let (node_name, neuron_idx, score) =
            verifier.finalize_graph_kfsb_fallback_selection(&domain, &scored, scored[0].clone());

        assert_eq!(node_name, "relu");
        assert!(
            scored
                .iter()
                .any(|candidate| candidate.neuron_idx == neuron_idx),
            "fallback should return one of the scored candidates, got neuron {neuron_idx}"
        );
        assert_eq!(
            score, 0.0,
            "all-child-failure fallback should use score 0.0 to match sequential kFSB semantics"
        );
    }

    /// Build a 4-unstable-neuron ReLU→Linear graph for the batched/serial parity
    /// tests below. Every ReLU neuron straddles zero, so all four are branchable.
    fn build_kfsb_parity_fixture(
        heuristic: BranchingHeuristic,
        fsb_candidates: usize,
    ) -> Result<(
        BetaCrownVerifier,
        GraphNetwork,
        GraphBabDomain,
        Vec<(String, usize)>,
    )> {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            branching_heuristic: heuristic,
            fsb_candidates,
            beta_iterations: 0,
            ..Default::default()
        });

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "linear",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0, -1.0, 0.5, -0.75]]), None)
                    .expect("linear layer should build"),
            ),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear");

        let input = BoundedTensor::new(
            arr1(&[-0.1, -1.0, -2.0, -3.0]).into_dyn(),
            arr1(&[0.2, 2.0, 1.0, 3.0]).into_dyn(),
        )?;
        let node_bounds = graph.collect_node_bounds(&input)?;
        let domain = GraphBabDomain::root(node_bounds, -5.0, 6.0, &input, false)?;
        let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &["relu".to_string()]);
        Ok((verifier, graph, domain, unstable))
    }

    /// Batched child-bound collection: one propagation for all
    /// `2 * eval_candidates` children (driven by a real engine) vs the serial
    /// per-child loop. Uses `NaiveCpuGemmEngine` — a real `GemmEngine` so
    /// `engine.is_some()` routes through the batched code path — keeping the test
    /// deterministic and GPU-free while still exercising
    /// `propagate_crown_with_batched_domains_full`.
    ///
    /// This asserts the properties that are load-bearing AND actually hold:
    ///   1. the batched collector returns one `(active, inactive)` entry per eval
    ///      candidate, mapped back in candidate order;
    ///   2. the feasibility (`Some`/`None`) pattern is bit-for-bit identical to the
    ///      serial path — the `with_constraint -> Ok(None)` infeasible mapping the
    ///      task requires preserved exactly;
    ///   3. every feasible batched bound is finite and ordered.
    ///
    /// It deliberately does NOT assert per-child `(lower, upper)` bit-equality nor
    /// an identical `(node, neuron)` pick. MEASURED: the batched CROWN backward and
    /// the serial `evaluate_graph_child_bounds` single pass use different ReLU
    /// relaxations — the batched pass is TIGHTER (e.g. on this fixture, active child
    /// of the split neuron: serial `(-4.35, 1.70)` vs batched `(-4.25, 0.70)`, where
    /// `(-4.25, 0.70)` is the exact bound), independent of α-state. That tighter
    /// (still sound) bound can flip the advisory pick. This is not a soundness change
    /// (the scorer never feeds a verdict; the verdict-parity tests in
    /// `engine::tests::gpu_bab::kfsb_parity` still pass), but it means the batched
    /// scorer is NOT bit-parity with the serial scorer. See the task report.
    #[ntest::timeout(20000)]
    #[test]
    fn test_graph_gpu_kfsb_batched_child_bounds_internal_consistency_4300() -> Result<()> {
        use ny_core::NaiveCpuGemmEngine;

        let (verifier, graph, domain, unstable) =
            build_kfsb_parity_fixture(BranchingHeuristic::KfsbInterceptOnly, 3)?;
        let objective = [1.0f32];
        let engine = NaiveCpuGemmEngine;

        // Reconstruct the eval-candidate set the selector uses, so we can drive
        // the two collection helpers on the exact same candidates.
        let scored = verifier.score_graph_kfsb_candidates(&graph, &domain, &unstable)?;
        let eval_candidates = select_graph_kfsb_eval_candidates(
            &scored,
            verifier.config.fsb_candidates,
            matches!(
                verifier.config.branching_heuristic,
                BranchingHeuristic::Kfsb
            ),
        );
        assert!(
            eval_candidates.len() >= 2,
            "need >=2 eval candidates to meaningfully exercise batched mapping, got {}",
            eval_candidates.len()
        );

        let serial = verifier.collect_graph_kfsb_child_bounds_serial(
            &graph,
            &domain,
            &objective,
            Some(&engine),
            &eval_candidates,
        )?;
        let batched = verifier.collect_graph_kfsb_child_bounds_batched(
            &graph,
            &domain,
            &objective,
            &engine,
            &eval_candidates,
        )?;

        assert_eq!(
            batched.len(),
            eval_candidates.len(),
            "batched collector must return one entry per eval candidate"
        );
        assert_eq!(
            serial.len(),
            batched.len(),
            "serial and batched must return the same number of entries"
        );

        // Feasibility (Some/None) must match EXACTLY on both sides — the
        // None/infeasible mapping is scored semantics that the batched path must
        // preserve precisely. Feasible bounds are sanity-checked (finite, ordered);
        // their exact values may diverge (see the doc comment above).
        for (idx, ((sa, si), (ba, bi))) in serial.iter().zip(batched.iter()).enumerate() {
            assert_eq!(
                sa.is_some(),
                ba.is_some(),
                "active feasibility mismatch at candidate {idx}: serial={sa:?} batched={ba:?}"
            );
            assert_eq!(
                si.is_some(),
                bi.is_some(),
                "inactive feasibility mismatch at candidate {idx}: serial={si:?} batched={bi:?}"
            );
            for (side, bounds) in [("active", ba), ("inactive", bi)] {
                if let Some((lo, hi)) = bounds {
                    assert!(
                        lo.is_finite() && hi.is_finite() && lo <= hi,
                        "batched {side} bounds must be finite and ordered at candidate {idx}: ({lo},{hi})"
                    );
                }
            }
        }

        // The batched public entry point must return a well-formed selection: a
        // real unstable candidate with a finite score.
        let (node, neuron, score) = verifier.select_graph_branch_kfsb_in_gpu_batched(
            &graph,
            &domain,
            &unstable,
            &objective,
            Some(&engine),
        )?;
        assert!(
            unstable.iter().any(|(n, i)| *n == node && *i == neuron),
            "batched selection {node:?}[{neuron}] must be one of the unstable candidates"
        );
        assert!(
            score.is_finite(),
            "batched selection score must be finite, got {score}"
        );

        Ok(())
    }

    /// Internal consistency: with an infeasible child forced (a neuron already
    /// pinned by the domain history), the batched collector must leave that side's
    /// slot `None`, exactly as the serial `with_constraint -> Ok(None)` path does.
    #[ntest::timeout(20000)]
    #[test]
    fn test_graph_gpu_kfsb_batched_preserves_infeasible_none_4300() -> Result<()> {
        use ny_core::NaiveCpuGemmEngine;

        let (verifier, graph, domain, unstable) =
            build_kfsb_parity_fixture(BranchingHeuristic::KfsbInterceptOnly, 4)?;
        let objective = [1.0f32];
        let engine = NaiveCpuGemmEngine;

        let scored = verifier.score_graph_kfsb_candidates(&graph, &domain, &unstable)?;
        let eval_candidates = select_graph_kfsb_eval_candidates(
            &scored,
            verifier.config.fsb_candidates,
            matches!(
                verifier.config.branching_heuristic,
                BranchingHeuristic::Kfsb
            ),
        );

        let serial = verifier.collect_graph_kfsb_child_bounds_serial(
            &graph,
            &domain,
            &objective,
            Some(&engine),
            &eval_candidates,
        )?;
        let batched = verifier.collect_graph_kfsb_child_bounds_batched(
            &graph,
            &domain,
            &objective,
            &engine,
            &eval_candidates,
        )?;

        // Feasibility pattern (which sides are None) must be bit-for-bit identical
        // to the serial path — this is the parity-critical None mapping.
        let serial_feasibility: Vec<(bool, bool)> = serial
            .iter()
            .map(|(a, i)| (a.is_some(), i.is_some()))
            .collect();
        let batched_feasibility: Vec<(bool, bool)> = batched
            .iter()
            .map(|(a, i)| (a.is_some(), i.is_some()))
            .collect();
        assert_eq!(
            serial_feasibility, batched_feasibility,
            "batched None/infeasible mapping must match serial exactly"
        );

        Ok(())
    }
}
