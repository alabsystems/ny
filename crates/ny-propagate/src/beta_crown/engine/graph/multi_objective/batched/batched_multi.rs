// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-batched domain processing for multi-objective graph BaB verification.

use ny_core::GemmEngine;
use rayon::prelude::*;
use tracing::debug;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::branching::GraphNeuronConstraint;
use crate::beta_crown::domain::{
    GraphCrownContext, MultiObjDomainWithUnstable, MultiObjectiveGraphBabDomain,
    MultiObjectiveTargets,
};
use crate::faer_parallelism::RayonTaskGuard;
use crate::GraphNetwork;

use super::super::super::super::domain_results::MultiObjectiveGraphDomainResult;
use super::super::super::super::BetaCrownVerifier;
use super::super::shared::{
    merge_pruned_cached_las, merge_pruned_objective_bounds,
    multi_objective_gpu_single_pass_enabled, prune_cached_las_for_targets,
    prune_verified_multi_objective_targets,
};
use super::children::collect_multi_objective_children;

/// Chunk width for the GPU single-pass lane (#w5-bab-throughput): the batched
/// adapter folds the whole chunk into ONE wide GPU pass per β iteration (see
/// `gpu_beta_optimize_wide`), so a wider chunk amortizes the fixed per-pass Metal
/// dispatch + buffer-alloc + device-wait overhead across more domains. Chunking
/// still bounds the overrun past the deadline to ~one chunk of β-iteration passes
/// (the deadline is re-checked between chunks). 2026-07-11 cifar100 A/B: 8→255,
/// 64→511 explored domains at the same 57s BaB budget; 64 banks the full ~8× wide
/// throughput while keeping the worst-case overrun to one 64-domain chunk (~a few
/// seconds, well inside the 5s scored grace). `NY_MO_GPU_CHUNK=<n>` overrides.
const MO_GPU_SINGLE_PASS_CHUNK: usize = 64;

/// #metaroom-chain-wide: env override for the GPU single-pass chunk width
/// (`NY_MO_GPU_CHUNK=<n>`, default [`MO_GPU_SINGLE_PASS_CHUNK`] — unchanged when
/// unset). An explicitly present malformed/zero/out-of-range value returns
/// `None`, which disables this experimental lane for the batch and routes every
/// child through the existing per-child fallback. On the WIDE batched lane the
/// whole chunk is ONE GPU pass per β iteration, so a wider chunk amortizes the
/// fixed per-pass cost across more domains (metaroom's 6cnn conv chains: 8 →
/// 32/64 packs 40 → 160/320 wide rows, still small for the device).
fn parse_mo_gpu_single_pass_chunk(raw: Option<&str>) -> Option<usize> {
    let Some(raw) = raw else {
        return Some(MO_GPU_SINGLE_PASS_CHUNK);
    };
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    raw.parse::<usize>().ok().filter(|&n| n > 0)
}

fn mo_gpu_single_pass_chunk() -> Option<usize> {
    match std::env::var("NY_MO_GPU_CHUNK") {
        Ok(raw) => parse_mo_gpu_single_pass_chunk(Some(&raw)),
        Err(std::env::VarError::NotPresent) => parse_mo_gpu_single_pass_chunk(None),
        Err(std::env::VarError::NotUnicode(_)) => None,
    }
}

/// Intersect fresh child objective bounds with the bounds inherited from the
/// parent domain (#w5-bab-throughput).
///
/// Sound: the child's sub-region is a subset of the parent's, so the parent's
/// per-objective interval also encloses the child's reachable objective values;
/// the per-objective intersection `[max(l), min(u)]` is a valid — and never
/// looser — enclosure. NaN in a fresh entry is preserved verbatim so the
/// existing NaN rejection in `update_bounds` (#2982) still fires. A numerically
/// inverted intersection (possible only from f32 slop between two sound
/// enclosures) keeps the fresh bound (sound; matches legacy behavior).
pub(super) fn tighten_child_bounds_with_parent(
    inherited: &[(f32, f32)],
    fresh: Vec<(f32, f32)>,
) -> Vec<(f32, f32)> {
    if inherited.len() != fresh.len() {
        return fresh;
    }
    fresh
        .into_iter()
        .zip(inherited.iter())
        .map(|((fl, fu), &(il, iu))| {
            if fl.is_nan() || fu.is_nan() {
                return (fl, fu);
            }
            let l = fl.max(il);
            let u = fu.min(iu);
            if l <= u {
                (l, u)
            } else {
                (fl, fu)
            }
        })
        .collect()
}

impl BetaCrownVerifier {
    /// Process a batch of multi-objective domains with GPU-batched CROWN computation.
    ///
    /// Similar to `process_graph_domains_batched_gpu` but handles multiple objectives.
    /// This batches the CROWN computation across all child domains to improve GPU utilization.
    ///
    /// Part of #3813: `cut_pool` is a read-only view of the current cutting
    /// planes. The batched path applies existing cuts during CROWN backward
    /// propagation but does not generate or merge new cuts — that happens in
    /// the outer BaB loop after batch results return.
    // Justification: batched multi-objective processing needs graph, domains,
    // relu nodes, objective/threshold slices, engine, and the read-only cut
    // pool together; splitting this signature further would just mirror a
    // temporary context struct without reducing the actual call surface.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn process_graph_domains_batched_gpu_multi_objective(
        &self,
        graph: &GraphNetwork,
        domains: &[&MultiObjectiveGraphBabDomain],
        relu_nodes: &[String],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: &dyn GemmEngine,
        cut_pool: Option<&GraphCutPool>,
        endgame: bool,
    ) -> Vec<MultiObjectiveGraphDomainResult> {
        // #phase-telemetry (dark, NY_PHASE_TELEMETRY=1, print-only): mark the
        // FIRST domain batch entering the resnet batched BaB lane, once per
        // process — the boundary where the root pipeline hands over to BaB
        // domain processing. Gate-off is a cached-bool load; the `Once` fires
        // only when the gate is on, so unset stays byte-identical.
        if crate::phase_telemetry::phase_telemetry_enabled() {
            static FIRST_BATCH: std::sync::Once = std::sync::Once::new();
            FIRST_BATCH
                .call_once(|| crate::phase_telemetry::phase_marker("bab-first-domain-batch start"));
        }
        if domains.is_empty() {
            return Vec::new();
        }

        // Pre-filter: separate already-verified, violation, and to-process domains
        let mut quick_results: std::collections::HashMap<usize, MultiObjectiveGraphDomainResult> =
            std::collections::HashMap::new();
        let mut domains_to_process: Vec<(usize, &MultiObjectiveGraphBabDomain)> = Vec::new();

        for (idx, domain) in domains.iter().enumerate() {
            // Quick verification check
            if domain.all_verified() {
                quick_results.insert(idx, MultiObjectiveGraphDomainResult::AlreadyVerified);
                continue;
            }

            // Quick violation check
            if domain.any_violated(thresholds, false) {
                quick_results.insert(idx, MultiObjectiveGraphDomainResult::Violation);
                continue;
            }

            domains_to_process.push((idx, domain));
        }

        if domains_to_process.is_empty() {
            return (0..domains.len())
                .map(|idx| {
                    quick_results.remove(&idx).unwrap_or_else(|| {
                        tracing::warn!(
                            "process_graph_domains_batched_gpu_multi_objective: missing quick_result for idx {} (#1993)",
                            idx
                        );
                        MultiObjectiveGraphDomainResult::PropagationFailure
                    })
                })
                .collect();
        }

        // Find unstable neurons for all domains in parallel
        let unstable_per_domain: Vec<(usize, Vec<(String, usize)>)> = domains_to_process
            .par_iter()
            .map(|(idx, domain)| {
                let unstable = self.find_unstable_graph_neurons_multi(graph, domain, relu_nodes);
                (*idx, unstable)
            })
            .collect();

        // Separate domains with/without unstable neurons
        let mut domains_with_unstable: Vec<MultiObjDomainWithUnstable<'_>> = Vec::new();

        // O(1) index from domain idx → domain ref, replacing a per-iteration
        // linear `.find()` over `domains_to_process` (was O(D²) for batch size D,
        // up to thousands of domains). `idx` is the unique `.enumerate()` index
        // assigned when `domains_to_process` was built, so each key maps to
        // exactly one domain — identical to the first-match `.find()` semantics.
        let domain_by_idx: std::collections::HashMap<usize, &MultiObjectiveGraphBabDomain> =
            domains_to_process.iter().map(|(i, d)| (*i, *d)).collect();

        for (idx, unstable) in unstable_per_domain {
            let Some(domain) = domain_by_idx.get(&idx).copied() else {
                tracing::warn!(
                    "process_graph_domains_batched_gpu_multi_objective: missing domain at idx {} while resolving unstable set (#1993)",
                    idx
                );
                quick_results.insert(idx, MultiObjectiveGraphDomainResult::PropagationFailure);
                continue;
            };

            if unstable.is_empty() {
                // No unstable neurons - compute final bounds
                let context = GraphCrownContext::new(
                    &domain.history,
                    cut_pool, // Part of #3813: apply existing cuts
                    Some(&domain.node_bounds),
                    Some(engine),
                )
                .with_alpha(&domain.alpha_state);
                match self.propagate_crown_with_graph_constraints(
                    graph,
                    domain.input_bounds.as_ref(),
                    &context,
                    None,
                    None, // Multi-objective: compute full output bounds
                ) {
                    Ok((output, _node_cache)) => {
                        match Self::objective_bounds_multi(&output, objectives) {
                            Ok(new_bounds) => {
                                // Defense-in-depth: reject length mismatch instead of
                                // silent .zip() truncation (#3383).
                                if new_bounds.len() != thresholds.len() {
                                    debug!(
                                        "batched multi-objective NoUnstable: new_bounds/thresholds length mismatch ({} vs {}) (#3383)",
                                        new_bounds.len(),
                                        thresholds.len()
                                    );
                                    quick_results.insert(
                                        idx,
                                        MultiObjectiveGraphDomainResult::PropagationFailure,
                                    );
                                    continue;
                                }
                                let all_verified = new_bounds
                                    .iter()
                                    .zip(thresholds.iter())
                                    .all(|((l, _), &t)| *l > t);
                                // #1866: Compute any_violated so the BaB loop can detect
                                // conclusive violations in fully-constrained domains.
                                let any_violated = new_bounds
                                    .iter()
                                    .zip(thresholds.iter())
                                    .any(|((_, u), &t)| *u < t);
                                quick_results.insert(
                                    idx,
                                    MultiObjectiveGraphDomainResult::NoUnstable {
                                        all_verified,
                                        any_violated,
                                    },
                                );
                            }
                            Err(e) => {
                                debug!(error = %e, "Multi-objective bounds extraction failed — returning PropagationFailure (#1978)");
                                quick_results.insert(
                                    idx,
                                    MultiObjectiveGraphDomainResult::PropagationFailure,
                                );
                            }
                        }
                    }
                    Err(ref e) if e.is_infeasible_domain() => {
                        // #2926: Infeasible domain = empty = trivially verified.
                        debug!(error = %e, "Multi-objective NoUnstable infeasible (empty)");
                        quick_results.insert(idx, MultiObjectiveGraphDomainResult::AlreadyVerified);
                    }
                    Err(e) => {
                        debug!(error = %e, "Multi-objective NoUnstable CROWN propagation failed — returning PropagationFailure (#1978)");
                        quick_results
                            .insert(idx, MultiObjectiveGraphDomainResult::PropagationFailure);
                    }
                }
            } else {
                domains_with_unstable.push((idx, domain, unstable));
            }
        }

        if domains_with_unstable.is_empty() {
            return (0..domains.len())
                .map(|idx| {
                    quick_results.remove(&idx).unwrap_or_else(|| {
                        tracing::warn!(
                            "process_graph_domains_batched_gpu_multi_objective: missing result for idx {} after unstable scan (#1993)",
                            idx
                        );
                        MultiObjectiveGraphDomainResult::PropagationFailure
                    })
                })
                .collect();
        }

        // #kfsb-multi (dark, NY_MO_KFSB=1): wave-batched kFSB branch selection.
        // Pre-scores + SIMULATES both children of the top-k∪backup candidates
        // for the whole wave in chunked dense-spec backward calls, picks per
        // domain by the configured reduce op on that domain's worst-straggler
        // row, and COMMITS the winner's already-built children (no rebuild).
        // Advisory-only (chooses WHICH neuron to split) ⇒ soundness-free; any
        // per-domain miss falls back to `select_graph_branch_multi` below.
        // Gate off ⇒ empty map ⇒ byte-identical to today.
        let kfsb_precomputed: std::sync::Mutex<
            std::collections::HashMap<usize, super::kfsb_multi::KfsbMultiChildren>,
        > = std::sync::Mutex::new(if self.kfsb_multi_wave_enabled() {
            self.select_graph_branch_kfsb_multi_batched(
                graph,
                &domains_with_unstable,
                relu_nodes,
                objectives,
                thresholds,
                engine,
            )
        } else {
            std::collections::HashMap::new()
        });

        // For domains with unstable neurons, create all children in parallel.
        // Returns Ok(children_info) on success, Err(idx) on branch selection failure (#2143).
        let child_creation_results: Vec<_> = domains_with_unstable
            .par_iter()
            .map(|(idx, domain, unstable)| {
                // #kfsb-multi: committed winner children, if the wave selector
                // resolved this domain (children_info shape matches the
                // advisory path: 0..=2 entries, infeasible halves absent).
                if let Some(pre) = kfsb_precomputed.lock().ok().and_then(|mut m| m.remove(idx)) {
                    let children_info: Vec<_> = pre
                        .into_iter()
                        .map(|(child, is_active)| (*idx, child, is_active))
                        .collect();
                    return Ok((*idx, children_info));
                }
                let (node_name, neuron_idx, score) = match self.select_graph_branch_multi(
                    graph,
                    domain,
                    unstable,
                    objectives,
                    Some(engine),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(
                            "select_graph_branch_multi failed for idx {}: {e} (#1915)",
                            idx
                        );
                        return Err(*idx);
                    }
                };

                let mut children_info = Vec::with_capacity(2);

                // Active child
                let active_constraint = GraphNeuronConstraint {
                    node_name: node_name.clone(),
                    neuron_idx,
                    is_active: true,
                    score,
                };
                match domain.with_constraint(graph, active_constraint, false, thresholds) {
                    Ok(Some(child)) => children_info.push((*idx, child, true)),
                    Ok(None) => {}
                    Err(ref e) if e.is_infeasible_domain() => {
                        // #2926: Infeasible constraint = empty child, skip.
                    }
                    Err(e) => {
                        tracing::warn!("with_constraint (active) failed for idx {}: {e}", idx);
                        return Err(*idx);
                    }
                }

                // Inactive child
                let inactive_constraint = GraphNeuronConstraint {
                    node_name,
                    neuron_idx,
                    is_active: false,
                    score,
                };
                match domain.with_constraint(graph, inactive_constraint, false, thresholds) {
                    Ok(Some(child)) => children_info.push((*idx, child, false)),
                    Ok(None) => {}
                    Err(ref e) if e.is_infeasible_domain() => {
                        // #2926: Infeasible constraint = empty child, skip.
                    }
                    Err(e) => {
                        tracing::warn!("with_constraint (inactive) failed for idx {}: {e}", idx);
                        return Err(*idx);
                    }
                }

                Ok((*idx, children_info))
            })
            .collect();

        // Handle branch selection failures explicitly (#2143).
        let mut branch_selection_failures: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let successful_results: Vec<_> = child_creation_results
            .into_iter()
            .filter_map(|result| match result {
                Ok(v) => Some(v),
                Err(failed_idx) => {
                    branch_selection_failures.insert(failed_idx);
                    quick_results.insert(
                        failed_idx,
                        MultiObjectiveGraphDomainResult::PropagationFailure,
                    );
                    None
                }
            })
            .collect();

        // Collect all children that need CROWN bounds computation.
        let (all_children, parent_domain_lookup) = collect_multi_objective_children(
            &domains_with_unstable,
            &successful_results,
            &mut quick_results,
        );

        // Per-child single-pass / beta-opt CROWN evaluation, unchanged from the
        // original closure. Used directly for the FALLBACK partition and as the
        // whole-batch fallback when the domain-batched primitive errors.
        let eval_child = |parent_idx: &usize, child: &MultiObjectiveGraphBabDomain| {
            let _rayon_task_guard = RayonTaskGuard::new();
            let Some(parent) = parent_domain_lookup.get(parent_idx) else {
                tracing::warn!(
                    "process_graph_domains_batched_gpu_multi_objective: missing parent lookup for child of idx {}",
                    parent_idx
                );
                return Err(false);
            };

            // Use beta-CROWN with SPSA optimization for shallow domains
            let mut beta_state = child.beta_state.clone();
            let context = GraphCrownContext::new(
                &child.history,
                cut_pool, // Part of #3813: apply existing cuts (read-only)
                Some(&parent.node_bounds),
                Some(engine),
            )
            .with_alpha(&child.alpha_state);
            let pruned_targets =
                prune_verified_multi_objective_targets(objectives, thresholds, &child.verified);
            let targets = MultiObjectiveTargets::new(
                &pruned_targets.objectives,
                &pruned_targets.thresholds,
                &pruned_targets.verified_mask,
            );
            let pruned_cached_las =
                prune_cached_las_for_targets(child.cached_las(), &pruned_targets);
            // Only run beta optimization when enabled and for shallow domains.
            let result =
                if self.config.beta_iterations > 0 && child.depth <= self.config.beta_max_depth {
                    self.optimize_graph_beta_analytical_multi_objective_with_cache(
                        graph,
                        child.input_bounds.as_ref(),
                        &context,
                        &mut beta_state,
                        &targets,
                        false, // conjunctive: batched path always uses disjunctive mode (#3334 closed)
                        &pruned_cached_las,
                        true,
                    )
                } else {
                    // Skip optimization, just propagate with inherited beta
                    self.propagate_multi_objective_with_beta_and_cache(
                        graph,
                        child.input_bounds.as_ref(),
                        &context,
                        &beta_state,
                        &targets,
                        &pruned_cached_las,
                        true,
                    )
                };
            match result {
                Ok((active_bounds, node_cache, active_cached_las)) => {
                    let obj_bounds = merge_pruned_objective_bounds(
                        &child.objective_bounds,
                        &pruned_targets,
                        active_bounds,
                    );
                    Ok((
                        obj_bounds,
                        node_cache,
                        beta_state,
                        // Per-child CPU path never persists ascent α — the
                        // child keeps its inherited α (#hard-six unshared-α).
                        None,
                        active_cached_las,
                        pruned_targets,
                    ))
                }
                // #2926: Preserve infeasibility signal through the parallel closure.
                Err(ref e) if e.is_infeasible_domain() => Err(true),
                Err(e) => {
                    tracing::warn!("Batched multi-objective child propagation failed: {e}");
                    Err(false)
                }
            }
        };

        // GPU single-pass lane (#w5-bab-throughput): when the engine provides the
        // sound GPU CROWN backward on a conv graph, route beta-opt-eligible
        // children through the domain-batched single-pass adapter too. Measured
        // (cifar100 prop_idx_7641, release): the per-child CPU beta-opt inner
        // pass costs ~3s (conv2d_transpose_backward_coeff_f64-dominated), so ONE
        // domain consumed the whole BaB window; the adapter's whole-suffix GPU
        // sound backward (try_gpu_beta_batched_resnet, alpha-bridged, inherited-β
        // dual folded) bounds a child in a fraction of that. Trades per-domain β
        // OPTIMIZATION for ~10x domain throughput. Default ON; NY_MO_GPU_BATCH=0
        // restores the legacy per-child beta-opt lane byte-identically.
        let gpu_single_pass_lane = multi_objective_gpu_single_pass_enabled()
            && graph.has_conv_layers()
            && (crate::sound_gpu_gate::global_sound_gpu_crown_for_wide().is_some()
                || engine
                    .as_gpu_crown_backward()
                    .is_some_and(|g| g.provides_sound_gpu_crown()));

        // Partition children into a DOMAIN-batchable single-pass set and a
        // FALLBACK set. A child is batchable iff ALL of:
        //   * cuts inactive (the dense-spec primitive does not apply cuts);
        //   * the single-pass branch applies (NOT beta-opt for this depth, OR
        //     the GPU single-pass lane is on);
        //   * no per-disjunct alphas (no GraphBabDomain equivalent);
        //   * every relu node is present in `child.node_bounds`
        //     (`from_graph_domains` errors otherwise);
        //   * the first objective bound is finite.
        // Everything else falls back to the EXACT per-child path.
        let cuts_inactive = cut_pool.map_or(true, |pool| pool.is_empty());
        let mut batchable_positions: Vec<usize> = Vec::new();
        for (pos, (_parent_idx, child, _is_active)) in all_children.iter().enumerate() {
            let single_pass_branch = gpu_single_pass_lane
                || !(self.config.beta_iterations > 0 && child.depth <= self.config.beta_max_depth);
            let relu_nodes_present = relu_nodes
                .iter()
                .all(|name| child.node_bounds.contains_key(name));
            let first_obj_finite = child
                .objective_bounds
                .first()
                .is_some_and(|(l, u)| l.is_finite() && u.is_finite());
            let batchable = cuts_inactive
                && single_pass_branch
                && child.per_disjunct_alphas().is_none()
                && relu_nodes_present
                && first_obj_finite;
            if batchable {
                batchable_positions.push(pos);
            }
        }

        // Run the FALLBACK set through the existing per-child path (in parallel).
        let mut child_bounds: Vec<Option<_>> = (0..all_children.len()).map(|_| None).collect();
        // O(1) membership for the fallback complement, replacing a linear
        // `batchable_positions.contains(pos)` inside a filter over all child
        // positions (was O(C²) for C children ≈ 2D). `batchable_positions`
        // holds distinct positions, so a HashSet yields identical membership.
        let batchable_set: std::collections::HashSet<usize> =
            batchable_positions.iter().copied().collect();
        let fallback_positions: Vec<usize> = (0..all_children.len())
            .filter(|pos| !batchable_set.contains(pos))
            .collect();
        let fallback_results: Vec<_> = fallback_positions
            .par_iter()
            .map(|&pos| {
                let (parent_idx, child, _is_active) = &all_children[pos];
                (pos, eval_child(parent_idx, child))
            })
            .collect();
        for (pos, result) in fallback_results {
            child_bounds[pos] = Some(result);
        }

        // Run the BATCHABLE set through the domain-batched single-pass adapter,
        // with per-chunk fallback to the per-child path on any batched error.
        //
        // Chunking (#w5-bab-throughput): with the GPU single-pass lane on, a BaB
        // batch can hold up to `batch_size` (256) children, each one whole-suffix
        // GPU backward — un-interruptible for minutes as a single call. Bounded
        // chunks with a deadline check between them cap the overrun; children
        // left unprocessed at the deadline surface as propagation failures
        // (sound: their sub-regions stay unexplored, the parent is unresolved,
        // and the outer loop returns Timeout). Legacy lane: one chunk, exactly
        // the previous single call.
        if !batchable_positions.is_empty() {
            let chunk_size = if gpu_single_pass_lane {
                mo_gpu_single_pass_chunk()
            } else {
                Some(batchable_positions.len())
            };
            if let Some(chunk_size) = chunk_size {
                let mut deadline_expired = false;
                // #endgame-grace (dark, NY_ENDGAME_GRACE_SECS, default 0 = off ⇒
                // byte-identical): when this batch is the ENTIRE remaining
                // frontier (`endgame`), finishing its chunks within a bounded
                // overrun past the α-deadline converts a fully-verifying tree
                // into `unsat` instead of an Unknown tainted by the dropped
                // tail. SOUND: deadlines only schedule work; the overrun eats
                // the post-BaB-PGD reserve, which is moot on a tree about to
                // fully verify (a late Violation still surfaces as `sat`). The
                // caller (CLI/preset) sizes the grace to its wall-budget layout.
                let grace = if endgame {
                    std::env::var("NY_ENDGAME_GRACE_SECS")
                        .ok()
                        .and_then(|v| v.parse::<f32>().ok())
                        .filter(|g| g.is_finite() && *g > 0.0)
                        .unwrap_or(0.0)
                } else {
                    0.0
                };
                for chunk in batchable_positions.chunks(chunk_size.max(1)) {
                    let past_soft = deadline_expired || self.config.alpha_config.past_deadline();
                    // Past the SOFT deadline: drop, unless the endgame grace is
                    // armed and the HARD (soft + grace) deadline has not passed.
                    let drop_chunk = past_soft
                        && (grace <= 0.0
                            || deadline_expired
                            || self.config.alpha_config.deadline.is_none_or(|d| {
                                std::time::Instant::now()
                                    >= d + std::time::Duration::from_secs_f32(grace)
                            }));
                    if drop_chunk {
                        if !deadline_expired
                            && std::env::var("NY_PROPFAIL_PROBE").ok().as_deref() == Some("1")
                        {
                            eprintln!(
                                "[propfail] site=deadline-drop chunk_dropped={}",
                                chunk.len()
                            );
                        }
                        deadline_expired = true;
                        for &pos in chunk {
                            child_bounds[pos] = Some(Err(false));
                        }
                        continue;
                    }
                    let chunk_refs: Vec<&MultiObjectiveGraphBabDomain> =
                        chunk.iter().map(|&pos| &all_children[pos].1).collect();
                    match self.batched_single_pass_multi_objective_children(
                        graph,
                        &chunk_refs,
                        relu_nodes,
                        objectives,
                        thresholds,
                        engine,
                        gpu_single_pass_lane,
                    ) {
                        Some(batched_results) => {
                            debug_assert_eq!(batched_results.len(), chunk.len());
                            for (&pos, result) in chunk.iter().zip(batched_results) {
                                // #nobranch-f64 (dark, NY_NOBRANCH_F64=1, default OFF
                                // => byte-identical): the batched dense-specs primitive
                                // can DROP a child (Err(false): a batch-level
                                // spec-row/union bookkeeping mismatch at
                                // batched_dense_specs.rs:387/406, or an f32 overflow) —
                                // a BATCH artifact, not a real failure. One such drop
                                // taints the whole run Unknown via has_unresolved even
                                // after the bound has CONVERGED (224 drops measured on
                                // cifar100 idx_9502). Retry the dropped child on the
                                // exact per-child `eval_child` path — the SAME sound
                                // backward with per-child spec bookkeeping — which
                                // succeeds where the batched primitive rejects.
                                let result = if matches!(result, Err(false))
                                    && matches!(
                                        std::env::var("NY_NOBRANCH_F64").ok().as_deref(),
                                        Some("1")
                                    ) {
                                    let (parent_idx, child, _) = &all_children[pos];
                                    eval_child(parent_idx, child)
                                } else {
                                    result
                                };
                                child_bounds[pos] = Some(result);
                            }
                        }
                        None => {
                            // Chunk fallback: route this chunk back through the
                            // exact per-child path (sound, mirrors batched_single.rs).
                            let fb: Vec<_> = chunk
                                .par_iter()
                                .map(|&pos| {
                                    let (parent_idx, child, _is_active) = &all_children[pos];
                                    (pos, eval_child(parent_idx, child))
                                })
                                .collect();
                            for (pos, result) in fb {
                                child_bounds[pos] = Some(result);
                            }
                        }
                    }
                }
            } else {
                tracing::warn!(
                    "invalid NY_MO_GPU_CHUNK; disabling the GPU single-pass lane for this batch"
                );
                let fb: Vec<_> = batchable_positions
                    .par_iter()
                    .map(|&pos| {
                        let (parent_idx, child, _is_active) = &all_children[pos];
                        (pos, eval_child(parent_idx, child))
                    })
                    .collect();
                for (pos, result) in fb {
                    child_bounds[pos] = Some(result);
                }
            }
        }

        // Every position must now be filled (batchable ∪ fallback == all).
        let child_bounds: Vec<_> = child_bounds
            .into_iter()
            .map(|slot| {
                slot.unwrap_or_else(|| {
                    tracing::warn!(
                        "process_graph_domains_batched_gpu_multi_objective: unfilled child bound slot (#partition)"
                    );
                    Err(false)
                })
            })
            .collect();

        // Build results from child bounds
        let mut children_by_parent: std::collections::HashMap<
            usize,
            Vec<(MultiObjectiveGraphBabDomain, bool)>,
        > = std::collections::HashMap::new();

        // #1861: Track parents that had child propagation failures or violations.
        let mut parents_with_failure: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let mut parents_with_violation: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        // #boxlift [frontier] telemetry (dark, NY_PHASE_TELEMETRY=1, print-only):
        // per-depth worst-child accumulator for THIS batch. The resnet batched
        // lane is otherwise `[converge]`-silent; the BOXLIFT decision table
        // (docs/BOXLIFT_CHARTER.md §2.1(a)) needs a per-depth worst-child metric.
        // Gate checked FIRST and cached here — gate-off is one bool load before
        // the loop plus one predictable branch per child (no map, no arithmetic,
        // no allocation). Read-only: nothing downstream reads the accumulator,
        // the counter, or the frames; verdicts are byte-identical either way.
        let frontier_telemetry_on = crate::phase_telemetry::phase_telemetry_enabled();
        let mut frontier_batch_domains: u64 = 0;
        let mut frontier_worst_by_depth: std::collections::BTreeMap<usize, f32> =
            std::collections::BTreeMap::new();

        for ((parent_idx, mut child, _is_active), bounds_result) in
            all_children.into_iter().zip(child_bounds)
        {
            match bounds_result {
                Ok((
                    obj_bounds,
                    node_cache,
                    beta_state,
                    alpha_state,
                    active_cached_las,
                    pruned_targets,
                )) => {
                    // #w5-bab-throughput: monotone BaB bound inheritance. The
                    // single-pass bound source can differ from the parent's
                    // (root bounds = margin ∩ GPU ∩ IBP); without this a child
                    // could REGRESS below the parent's already-proven bound and
                    // stall convergence. `child.objective_bounds` still holds
                    // the inherited parent bounds here (update_bounds not yet
                    // called). Applied only on the GPU lane so the legacy flag
                    // stays byte-identical.
                    let obj_bounds = if gpu_single_pass_lane {
                        tighten_child_bounds_with_parent(&child.objective_bounds, obj_bounds)
                    } else {
                        obj_bounds
                    };
                    // #cone-delta increment 2: the result map is already
                    // Arc-shared — install by move (the historical re-Arc of
                    // every tensor is gone).
                    child.node_bounds = node_cache;
                    // #cone-delta: post-bounding replacement — delta restarts
                    // empty.
                    child.delta_pre_nodes.clear();
                    child.beta_state = beta_state;
                    // #hard-six unshared-α (dark, NY_WIDE_ALPHA_UNSHARED=1):
                    // persist the wide ascent's best-margin α onto the child so
                    // its descendants inherit the ascended per-neuron slopes
                    // via `from_parent`. `None` (gate off / non-participant /
                    // CPU path) keeps the inherited α — byte-identical.
                    if let Some(alpha_state) = alpha_state {
                        child.alpha_state = alpha_state;
                    }
                    let mut bounds_ok = child.update_bounds(obj_bounds, thresholds, false).is_ok();
                    if !bounds_ok
                        && matches!(std::env::var("NY_NOBRANCH_F64").ok().as_deref(), Some("1"))
                    {
                        // #nobranch-f64 (dark, NY_NOBRANCH_F64=1, default OFF =>
                        // byte-identical): the GPU sound-f32 wide-β backward produced a
                        // NON-FINITE (NaN/inf) objective bound for this child — an f32
                        // overflow in the error combine, NOT a real infeasibility. One
                        // such child taints an otherwise-converged run as Unknown via
                        // has_unresolved(propagation_failure). Recompute the child's
                        // bound on the CPU f64 sound backward (engine=None): larger
                        // range, no f32 overflow. SOUND: f64 is a valid tighter
                        // enclosure and update_bounds keeps its own NaN guard.
                        let f64_context = GraphCrownContext::new(
                            &child.history,
                            cut_pool,
                            Some(&child.node_bounds),
                            None,
                        )
                        .with_alpha(&child.alpha_state);
                        if let Ok((f64_output, _)) = self.propagate_crown_with_graph_constraints(
                            graph,
                            child.input_bounds.as_ref(),
                            &f64_context,
                            None,
                            None,
                        ) {
                            if let Ok(f64_bounds) =
                                Self::objective_bounds_multi(&f64_output, objectives)
                            {
                                bounds_ok =
                                    child.update_bounds(f64_bounds, thresholds, false).is_ok();
                            }
                        }
                    }
                    if !bounds_ok {
                        // NaN in objective bounds → treat as propagation failure (#2982)
                        if std::env::var("NY_PROPFAIL_PROBE").ok().as_deref() == Some("1") {
                            eprintln!("[propfail] site=NaN-bounds depth={}", child.depth);
                        }
                        parents_with_failure.insert(parent_idx);
                        continue;
                    }

                    let all_verified = child.all_verified();
                    let any_violated = child.any_violated(thresholds, false);

                    // #boxlift [frontier] hook: fold this child into the batch's
                    // per-depth worst-unverified-margin frame. depth = the
                    // child's split count (`depth` increments once per split
                    // constraint, so it IS the split_count); margin = lb − t on
                    // an unverified objective (this lane runs lower-bound mode:
                    // update_bounds/any_violated above pass verify_upper=false).
                    // Only surviving frontier children (neither all-verified nor
                    // violated) contribute a margin; every child that completed
                    // a bounds update counts toward the cumulative domain
                    // counter. Bounds here are finite: update_bounds rejects
                    // non-finite unverified bounds via its priority fold (#2982).
                    if frontier_telemetry_on {
                        frontier_batch_domains += 1;
                        if !all_verified && !any_violated {
                            let mut child_worst = f32::INFINITY;
                            for (((lb, _), &t), &v) in child
                                .objective_bounds
                                .iter()
                                .zip(thresholds.iter())
                                .zip(child.verified.iter())
                            {
                                if !v {
                                    child_worst = child_worst.min(lb - t);
                                }
                            }
                            if child_worst < f32::INFINITY {
                                let slot = frontier_worst_by_depth
                                    .entry(child.depth)
                                    .or_insert(f32::INFINITY);
                                *slot = slot.min(child_worst);
                            }
                        }
                    }

                    if any_violated {
                        // #1861: Track violated children instead of silently dropping.
                        parents_with_violation.insert(parent_idx);
                    } else {
                        let merged_cached_las = merge_pruned_cached_las(
                            child.cached_las(),
                            &pruned_targets,
                            active_cached_las,
                        );
                        if child.set_cached_las(merged_cached_las).is_err() {
                            if std::env::var("NY_PROPFAIL_PROBE").ok().as_deref() == Some("1") {
                                eprintln!("[propfail] site=cached_las depth={}", child.depth);
                            }
                            parents_with_failure.insert(parent_idx);
                            continue;
                        }
                        children_by_parent
                            .entry(parent_idx)
                            .or_default()
                            .push((child, all_verified));
                    }
                }
                Err(true) => {
                    // #2926: Infeasible domain = empty = trivially verified.
                    // Ensure parent doesn't fall to PropagationFailure when both children infeasible.
                    children_by_parent.entry(parent_idx).or_default();
                }
                Err(false) => {
                    // #1861: child bounds computation failed — sub-region unexplored.
                    if std::env::var("NY_PROPFAIL_PROBE").ok().as_deref() == Some("1") {
                        eprintln!("[propfail] site=child-eval-Err-false parent={parent_idx}");
                    }
                    parents_with_failure.insert(parent_idx);
                }
            }
        }

        // #boxlift [frontier] emission: at most ONE line per distinct depth per
        // BATCH (batch-level frames, not per-domain — a mixed-depth batch gets
        // one line per depth present, in depth order via the BTreeMap). The
        // cumulative domain counter is a process-wide atomic (function-local
        // `static`, same idiom as FIRST_BATCH above); it only advances when the
        // gate is on, and nothing but this print ever reads it. Gate-off skips
        // the whole block — byte-identical, zero output.
        if frontier_telemetry_on && frontier_batch_domains > 0 {
            static FRONTIER_DOMAINS_CUMULATIVE: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let cumulative = FRONTIER_DOMAINS_CUMULATIVE
                .fetch_add(frontier_batch_domains, std::sync::atomic::Ordering::Relaxed)
                + frontier_batch_domains;
            for (&depth, &worst) in &frontier_worst_by_depth {
                crate::phase_telemetry::frontier_frame(depth, worst, cumulative);
            }
        }

        // NY_BRANCH_TRACE (dark, diagnostic-only): per-split frontier-bound lift.
        // For every parent that produced children, log the chosen split
        // (node,neuron), the parent's worst-unverified straggler LB, the
        // post-split bound on THAT objective (= min over the split's children of
        // their LB on the straggler row — the domain's effective bound after the
        // partition), and the lift. Aggregating these lines answers "does each
        // split move the frontier bound, or are most splits wasted?" and "is the
        // same layer repeatedly chosen?". Print-only; gate off ⇒ the whole block
        // is skipped ⇒ byte-identical. Advisory measurement ⇒ soundness-free.
        if std::env::var("NY_BRANCH_TRACE").ok().as_deref() == Some("1") {
            let unstable_count: std::collections::HashMap<usize, usize> = domains_with_unstable
                .iter()
                .map(|(i, _, u)| (*i, u.len()))
                .collect();
            for (parent_idx, children_vec) in &children_by_parent {
                let Some(parent) = parent_domain_lookup.get(parent_idx) else {
                    continue;
                };
                // Worst unverified straggler on the parent (mirrors the selector).
                let mut straggler: Option<(usize, f32)> = None;
                for (i, (lo, _)) in parent.objective_bounds.iter().enumerate() {
                    if parent.verified.get(i).copied().unwrap_or(false) {
                        continue;
                    }
                    let lo = if lo.is_nan() { f32::NEG_INFINITY } else { *lo };
                    if straggler.is_none_or(|(_, w)| lo < w) {
                        straggler = Some((i, lo));
                    }
                }
                let Some((s_idx, parent_lb)) = straggler else {
                    continue;
                };
                // Post-split bound = min over children of the child's LB on the
                // straggler objective (the partition's effective bound on `s_idx`).
                let mut post = f32::INFINITY;
                for (child, _) in children_vec {
                    if let Some((lo, _)) = child.objective_bounds.get(s_idx) {
                        let lo = if lo.is_nan() { f32::NEG_INFINITY } else { *lo };
                        post = post.min(lo);
                    }
                }
                // The chosen split = the newest constraint on a child.
                let (node, neuron) = children_vec
                    .first()
                    .and_then(|(c, _)| c.history.constraints.last())
                    .map(|c| (c.node_name.clone(), c.neuron_idx))
                    .unwrap_or_else(|| ("?".to_string(), usize::MAX));
                eprintln!(
                    "[branch-trace] depth={} node={} neuron={} straggler={} parent_lb={:.5} post_lb={:.5} lift={:.5} nchild={} nunstable={}",
                    parent.depth,
                    node,
                    neuron,
                    s_idx,
                    parent_lb,
                    post,
                    post - parent_lb,
                    children_vec.len(),
                    unstable_count.get(parent_idx).copied().unwrap_or(0),
                );
            }
        }

        // Assemble final results
        for (parent_idx, _, _) in &domains_with_unstable {
            // Branch selection failures are already in quick_results (#2143).
            if branch_selection_failures.contains(parent_idx) {
                continue;
            }
            if parents_with_failure.contains(parent_idx) {
                // #1861: propagation failure — soundness requires we flag this.
                quick_results.insert(
                    *parent_idx,
                    MultiObjectiveGraphDomainResult::PropagationFailure,
                );
            } else if parents_with_violation.contains(parent_idx) {
                // #1861: child violated — track as violation instead of silently dropping.
                quick_results.insert(*parent_idx, MultiObjectiveGraphDomainResult::Violation);
            } else if let Some(children) = children_by_parent.remove(parent_idx) {
                quick_results.insert(
                    *parent_idx,
                    MultiObjectiveGraphDomainResult::Children(children),
                );
            } else {
                // Both children infeasible (with_constraint returned None for both).
                // This is a legitimate outcome, not an internal failure (#2143).
                tracing::debug!(
                    "process_graph_domains_batched_gpu_multi_objective: both children infeasible for parent idx {} (#2143)",
                    parent_idx
                );
                quick_results.insert(
                    *parent_idx,
                    MultiObjectiveGraphDomainResult::PropagationFailure,
                );
            }
        }

        // Return results in order
        (0..domains.len())
            .map(|idx| {
                quick_results.remove(&idx).unwrap_or_else(|| {
                    tracing::warn!(
                        "process_graph_domains_batched_gpu_multi_objective: missing final result for idx {} (#1993)",
                        idx
                    );
                    MultiObjectiveGraphDomainResult::PropagationFailure
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod monotone_merge_tests {
    use super::tighten_child_bounds_with_parent;

    /// Sub-region argument (#w5-bab-throughput): the child's region is a subset
    /// of the parent's, so per-objective intersection with the inherited bounds
    /// is sound and never looser. Fresh-tighter, parent-tighter, and mixed
    /// entries must each resolve to the elementwise-tightest interval.
    #[test]
    fn tighten_child_bounds_takes_elementwise_tightest_w5() {
        let inherited = [(-0.06_f32, 5.0_f32), (-3.0, 2.0), (-1.0, 1.0)];
        let fresh = vec![(-0.5_f32, 4.0_f32), (-2.5, 3.0), (-1.0, 1.0)];
        let merged = tighten_child_bounds_with_parent(&inherited, fresh);
        // obj0: parent lower (-0.06) beats fresh (-0.5); fresh upper (4.0) beats parent.
        assert_eq!(merged[0], (-0.06, 4.0));
        // obj1: fresh lower tighter, parent upper tighter.
        assert_eq!(merged[1], (-2.5, 2.0));
        // obj2: identical stays identical.
        assert_eq!(merged[2], (-1.0, 1.0));
    }

    /// NaN in a fresh entry must survive verbatim so `update_bounds` (#2982)
    /// still rejects the child as numerically corrupted — silently masking a
    /// failed pass with the parent's bound would hide the corruption.
    #[test]
    fn tighten_child_bounds_preserves_fresh_nan_w5() {
        let inherited = [(0.0_f32, 1.0_f32)];
        let fresh = vec![(f32::NAN, 1.0_f32)];
        let merged = tighten_child_bounds_with_parent(&inherited, fresh);
        assert!(merged[0].0.is_nan(), "NaN must propagate to update_bounds");
    }

    /// A numerically inverted intersection (only possible from f32 slop between
    /// two sound enclosures) keeps the fresh bound — matching what the legacy
    /// lane would have reported.
    #[test]
    fn tighten_child_bounds_keeps_fresh_on_inverted_intersection_w5() {
        let inherited = [(0.5_f32, 0.6_f32)];
        let fresh = vec![(0.7_f32, 0.9_f32)];
        let merged = tighten_child_bounds_with_parent(&inherited, fresh);
        assert_eq!(merged[0], (0.7, 0.9));
    }

    /// Length mismatch (defensive) returns the fresh bounds unchanged.
    #[test]
    fn tighten_child_bounds_length_mismatch_returns_fresh_w5() {
        let inherited = [(0.0_f32, 1.0_f32)];
        let fresh = vec![(0.1_f32, 0.9_f32), (0.2, 0.8)];
        let merged = tighten_child_bounds_with_parent(&inherited, fresh.clone());
        assert_eq!(merged, fresh);
    }
}

#[cfg(test)]
mod mo_gpu_chunk_tests {
    use super::{parse_mo_gpu_single_pass_chunk, MO_GPU_SINGLE_PASS_CHUNK};

    #[test]
    fn unset_uses_the_unchanged_default() {
        assert_eq!(
            parse_mo_gpu_single_pass_chunk(None),
            Some(MO_GPU_SINGLE_PASS_CHUNK)
        );
    }

    #[test]
    fn positive_native_usize_values_are_accepted() {
        assert_eq!(parse_mo_gpu_single_pass_chunk(Some("128")), Some(128));
        assert_eq!(parse_mo_gpu_single_pass_chunk(Some("00128")), Some(128));
        let native_max = usize::MAX.to_string();
        assert_eq!(
            parse_mo_gpu_single_pass_chunk(Some(&native_max)),
            Some(usize::MAX)
        );
    }

    #[test]
    fn explicitly_invalid_values_disable_the_experimental_lane() {
        for raw in ["", "0", "+64", "-1", " 64", "64 ", "64.0"] {
            assert_eq!(
                parse_mo_gpu_single_pass_chunk(Some(raw)),
                None,
                "{raw:?} must fail closed"
            );
        }
        let overflow = (usize::MAX as u128 + 1).to_string();
        assert_eq!(parse_mo_gpu_single_pass_chunk(Some(&overflow)), None);
    }
}
