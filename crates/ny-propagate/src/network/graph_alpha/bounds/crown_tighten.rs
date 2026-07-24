// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core CROWN-IBP per-node tightening loop (#3596, #3499).

use crate::types::{
    BoundsProvenance, CrownIbpFallbackEvent, CrownIbpFallbackReason, GraphCrownIbpBoundsResult,
};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info};

use super::budget_policy;
use super::demand::nodes_requiring_crown_tightening;
use super::target_backward::{crown_cut_segment_from_env, CrownCutContext};
use crate::network::core::{GraphNetwork, GraphTargetShapeContract};

impl GraphNetwork {
    /// Core CROWN-IBP loop with optional width-based and per-node time skips.
    ///
    /// Reads the #crown-cut-segment gate (`NY_CROWN_CUT_SEGMENT`) once per
    /// collection and delegates to the explicit-segment variant below.
    pub(in crate::network::graph_alpha) fn collect_crown_ibp_bounds_core_inner(
        &self,
        input: &BoundedTensor,
        ibp_bounds: HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
        engine: Option<&dyn ny_core::GemmEngine>,
        min_width_to_tighten: Option<f32>,
    ) -> Result<GraphCrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_core_inner_with_cut_segment(
            input,
            ibp_bounds,
            deadline,
            engine,
            min_width_to_tighten,
            crown_cut_segment_from_env(),
        )
    }

    /// Explicit-cut-segment variant of the core loop (#crown-cut-segment).
    /// `cut_segment = 0` disables cuts (byte-identical full-prefix backward).
    /// Production always enters through the env-reading wrapper above; the
    /// soundness oracle injects the segment directly so it never mutates the
    /// process-global environment that cargo's parallel test threads share.
    pub(crate) fn collect_crown_ibp_bounds_core_inner_with_cut_segment(
        &self,
        input: &BoundedTensor,
        ibp_bounds: HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
        engine: Option<&dyn ny_core::GemmEngine>,
        min_width_to_tighten: Option<f32>,
        cut_segment: usize,
    ) -> Result<GraphCrownIbpBoundsResult> {
        let exec_order = self.exec_order()?;

        // Linear-chain graphs can reuse the sequential `Network` CROWN-IBP
        // collector, which already contains the #3599 GPU partial-backward
        // fast path and deadline support. The sequential collector checks
        // deadlines between each layer, matching the graph-native loop.
        //
        // Gates:
        // 1. Engine presence: without GPU, the graph-native loop provides
        //    better per-node fallback for unsupported layers.
        // 2. Width-threshold mode remains graph-specific.
        // 3. Skip-fraction: the graph path skips nodes whose identity
        //    matrix exceeds the CPU dense budget. When the majority of
        //    nodes would be skipped (>50%), the graph path's selectivity
        //    saves more work than the sequential GPU path's single-pass
        //    efficiency. When most nodes are within budget, sequential GPU
        //    wins because one backward pass is cheaper than O(N²) per-node
        //    backward passes on CPU.
        //
        //    Measured heuristic (#3599):
        //    - soundnessbench: 37% skip → sequential GPU 30x faster
        //    - metaroom: 69% skip → graph path 4.8x faster
        let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let total_count = ibp_bounds.len().max(1);
        // Count every dense-overflow target for the fast-path gate. The
        // sequential collector cannot selectively skip large spatial targets,
        // so metaroom must stay on the graph-native collector when those nodes
        // dominate. The graph-native loop below uses a separate, patches-aware
        // budget helper because it can still tighten those targets via #3813.
        let exceed_count = ibp_bounds
            .iter()
            .filter(|(_node_name, bound)| {
                Self::counts_toward_sequential_skip_fraction(bound, budget)
            })
            .count();
        let skip_fraction = exceed_count as f64 / total_count as f64;
        let majority_skipped = skip_fraction > 0.5;
        debug!(
            "CROWN-IBP collection fast-path: {exceed_count}/{total_count} nodes exceed budget \
             (skip_fraction={skip_fraction:.2}), majority_skipped={majority_skipped}, \
             engine={}",
            engine.is_some(),
        );
        if engine.is_some() && min_width_to_tighten.is_none() && !majority_skipped {
            if let Some(result) = self.try_collect_crown_ibp_bounds_via_sequential_network(
                exec_order,
                input,
                &ibp_bounds,
                engine,
                deadline,
            )? {
                return Ok(result);
            }
        }

        // #crown-cut-segment (NY_CROWN_CUT_SEGMENT, default OFF = full-prefix
        // backward): designate every N-th node of the execution order as a
        // CUT. A per-target backward that reaches a cut node whose bounds this
        // sweep already finalized concretizes the accumulated linear relation
        // against that node's box (same directed-rounding concretization as
        // the input box; see CROWN_CUT_SEGMENT_ENV in target_backward.rs)
        // instead of expanding the node's prefix, dropping the sweep from
        // O(n²) to ~O(n·N) backward steps. Bounds can only get LOOSER (still
        // sound): the map under construction only ever holds valid enclosures,
        // so every cut box is a valid enclosure by construction. Topological
        // order finalizes every ancestor before its dependents; a cut node the
        // map does not (yet) cover is simply expanded as usual (fail-open).
        let cut_ctx: Option<CrownCutContext> = (cut_segment > 0).then(|| {
            CrownCutContext::new(
                exec_order
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| index % cut_segment == 0)
                    .map(|(_, name)| name.clone())
                    .collect(),
            )
        });

        // #margin-subset-seed: resolve the OUTPUT node once. When the
        // relu-split initial-bounds scope published spec-referenced margin
        // indices (crate::output_margin_seed), the OUTPUT-node tightening
        // below seeds only those k identity rows and scatters them over the
        // node's sound IBP bounds instead of running the full
        // `[output_dim x output_dim]` identity backward.
        let margin_subset_output_node: Option<String> = if self.output_name().is_empty() {
            exec_order.last().cloned()
        } else {
            Some(self.output_name().to_string())
        };

        let mut crown_ibp_bounds: HashMap<String, BoundedTensor> = HashMap::new();
        let mut provenance: HashMap<String, BoundsProvenance> = HashMap::new();
        let mut fallback_events = Vec::new();
        let demand_set = nodes_requiring_crown_tightening(self, exec_order, &ibp_bounds);
        debug!(
            "CROWN-IBP demand set: {}/{} nodes selected",
            demand_set.len(),
            ibp_bounds.len()
        );
        let mut patches_budget = budget_policy::PatchesTighteningBudget::new();
        // Preset-configurable floor/cap for the equal-share per-node budget
        // (#cgan-bn11-budget). Default (all-None) reproduces the #3499/#4413
        // constants exactly; only presets that set the knobs (cgan_2023's
        // 150 s cap for the 28,800-dim BN_11 chunked backward) change policy.
        let per_node_time_budget = self.crown_ibp_per_node_time_budget;
        let (per_node_floor_secs, _) =
            budget_policy::effective_per_node_time_budget(&per_node_time_budget);
        // Per-node time-budget candidates. This mask only feeds
        // `count_remaining_budget_candidates` (the equal-share split of the
        // remaining deadline). Nodes whose dense identity exceeds the memory
        // budget now COUNT as candidates (#cgan-bn11-chunk): they are no longer
        // skipped to IBP but rerouted through the objective-chunked backward,
        // so they consume a per-node time share like any other CROWN target.
        let global_budget_candidate_mask: Vec<bool> = exec_order
            .iter()
            .map(|node_name| {
                let Some(ibp_bound) = ibp_bounds.get(node_name) else {
                    return false;
                };
                let width_eligible = min_width_to_tighten
                    .map(|threshold| ibp_bound.max_width() >= threshold)
                    .unwrap_or(true);
                demand_set.contains(node_name) && width_eligible
            })
            .collect();
        // COST-WEIGHTED budget (#cgan-collection-cost-weight): each candidate's
        // per-node time slice is proportional to its objective-row count
        // (`ibp_bound.len()`) rather than an equal split, so a wide generator
        // target (BatchNorm_11 = 28,800 dims) gets enough time to COMPLETE on the
        // first pass instead of degrading to IBP and forcing redundant
        // re-collections. Non-candidates carry weight 0.0.
        let global_budget_candidate_weights: Vec<f64> = exec_order
            .iter()
            .zip(global_budget_candidate_mask.iter())
            .map(|(node_name, &is_candidate)| {
                if !is_candidate {
                    return 0.0;
                }
                ibp_bounds
                    .get(node_name)
                    .map(|b| b.len() as f64)
                    .unwrap_or(0.0)
            })
            .collect();
        let mut deadline_exceeded = false;
        let total_nodes = exec_order.len();
        let collection_start = Instant::now();
        let mut crown_node_count = 0usize;
        let mut crown_total_secs = 0.0f64;
        let mut skip_count = 0usize;
        let mut demand_skip_count = 0usize;

        for (layer_index, node_name) in exec_order.iter().enumerate() {
            // Deadline check (#3109): if deadline exceeded, skip CROWN backward
            // for remaining nodes and use IBP bounds instead. This is sound
            // (IBP bounds are valid, just looser).
            if !deadline_exceeded {
                if let Some(d) = deadline {
                    if Instant::now() >= d {
                        info!(
                            "CROWN-IBP DAG: deadline exceeded at node {}/{}, remaining nodes use IBP",
                            layer_index, total_nodes
                        );
                        deadline_exceeded = true;
                    }
                }
            }

            let ibp_bound = match ibp_bounds.get(node_name) {
                Some(b) => b,
                None => continue,
            };
            let layer_type = self
                .nodes
                .get(node_name)
                .map(|node| node.layer.layer_type().to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            // Demand-driven skip (#3775): no downstream nonlinear consumer needs this node.
            if !demand_set.contains(node_name) {
                demand_skip_count += 1;
                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                provenance.insert(
                    node_name.clone(),
                    BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DemandDrivenSkip),
                );
                continue;
            }

            // When deadline exceeded, skip CROWN backward and use IBP directly (#3109).
            if deadline_exceeded {
                skip_count += 1;
                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                provenance.insert(
                    node_name.clone(),
                    BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded),
                );
                fallback_events.push(CrownIbpFallbackEvent {
                    layer_index,
                    layer_type,
                    reason: CrownIbpFallbackReason::DeadlineExceeded,
                    details: format!("node '{}' skipped CROWN (deadline exceeded)", node_name),
                });
                continue;
            }

            // Memory budget check: the graph-native loop may keep spatial unary
            // targets on the patches-start path from #3813, so only dense-only
            // targets trip this gate. This intentionally differs from the
            // sequential fast-path gate above, which must remain conservative
            // enough to keep metaroom off the wrong collector (#3839).
            //
            // Over-budget targets are no longer skipped to IBP: they reroute
            // through the bound-equivalent objective-chunked backward
            // (#cgan-bn11-chunk, `propagate_crown_to_node_chunked`) with an
            // auto chunk size that scales the identity pair down to the budget.
            // Memory is bounded by the chunk; a slow chunked node still
            // degrades to IBP via `per_node_deadline` (sound either way). The
            // under-budget path is unchanged (`chunk_override = None` keeps the
            // env-driven single-pass behavior byte-for-byte).
            let chunk_override =
                if self.graph_native_target_exceeds_budget(node_name, ibp_bound, budget) {
                    let node_dim = ibp_bound.len();
                    let required = crate::network::crown_memory::identity_pair_bytes(node_dim)
                        .unwrap_or(usize::MAX);
                    let chunk_rows = budget_policy::auto_objective_chunk_rows(node_dim, budget);
                    debug!(
                        "CROWN-IBP DAG: node '{}' dim={} identity requires {} bytes \
                     (budget {}), budget exceeded -> chunked backward C={}",
                        node_name, node_dim, required, budget, chunk_rows
                    );
                    Some(chunk_rows)
                } else {
                    None
                };
            let is_patches_target =
                self.crown_ibp_target_can_start_in_patches(node_name, ibp_bound);

            if is_patches_target
                && !patches_budget.can_start_node(budget_policy::MIN_PER_NODE_BUDGET_SECS)
            {
                let node_dim = ibp_bound.len();
                let patches_budget_used_secs = patches_budget.used_secs();
                debug!(
                    "CROWN-IBP DAG: node '{}' dim={} patches-eligible but aggregate \
                     patches budget exhausted/below {:.1}s floor ({patches_budget_used_secs:.3}s used), using IBP",
                    node_name,
                    node_dim,
                    budget_policy::MIN_PER_NODE_BUDGET_SECS,
                );
                skip_count += 1;
                budget_policy::record_patches_budget_fallback(
                    &mut crown_ibp_bounds,
                    &mut provenance,
                    &mut fallback_events,
                    node_name,
                    ibp_bound,
                    layer_index,
                    &layer_type,
                    node_dim,
                    patches_budget_used_secs,
                );
                continue;
            }

            // Width-based skip: when the IBP interval is already tight, CROWN
            // backward cannot meaningfully tighten further.  Skipping saves the
            // ~5-7s per-node cost for the budget to reach deeper, wider nodes
            // where tightening matters most (#3499).
            if let Some(threshold) = min_width_to_tighten {
                let ibp_max_width = ibp_bound.max_width();
                if ibp_max_width < threshold {
                    skip_count += 1;
                    debug!(
                        "CROWN-IBP DAG: node '{}' max_width={:.6} < threshold={:.6}, skipping CROWN",
                        node_name, ibp_max_width, threshold
                    );
                    crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                    provenance.insert(
                        node_name.clone(),
                        BoundsProvenance::ForwardFallback(
                            CrownIbpFallbackReason::WidthBelowThreshold,
                        ),
                    );
                    fallback_events.push(CrownIbpFallbackEvent {
                        layer_index,
                        layer_type,
                        reason: CrownIbpFallbackReason::WidthBelowThreshold,
                        details: format!(
                            "node '{}' max_width={:.6} < threshold={:.6}",
                            node_name, ibp_max_width, threshold
                        ),
                    });
                    continue;
                }
            }

            // Share the remaining deadline across the remaining globally
            // eligible tightening targets, then clamp with the #4413 cap.
            let global_per_node = deadline.and_then(|d| {
                let now = Instant::now();
                if now >= d {
                    return None;
                }
                let remaining = d.duration_since(now);
                let remaining_secs = remaining.as_secs_f64();
                // #cgan-collection-cost-weight: cost-proportional slice by
                // objective-row count so the wide generator target completes on
                // the first pass. Reduces to equal-share when all weights match.
                let remaining_weight_sum = budget_policy::sum_remaining_budget_weights(
                    &global_budget_candidate_weights,
                    layer_index,
                );
                let this_weight = global_budget_candidate_weights
                    .get(layer_index)
                    .copied()
                    .unwrap_or(0.0);
                let per_node_secs = budget_policy::compute_weighted_per_node_budget_secs(
                    remaining_secs,
                    remaining_weight_sum,
                    this_weight,
                    &per_node_time_budget,
                )?;
                Some(now + Duration::from_secs_f64(per_node_secs))
            });
            let patches_per_node = patches_budget
                .remaining_deadline(is_patches_target, budget_policy::MIN_PER_NODE_BUDGET_SECS);
            let per_node_deadline = budget_policy::merge_per_node_deadlines(
                global_per_node,
                patches_per_node,
                deadline.is_some(),
            );

            // Skip this node when its per-node share falls below the minimum
            // floor even though the global deadline has not expired yet.
            if deadline.is_some() && per_node_deadline.is_none() {
                let remaining_global_candidates = budget_policy::count_remaining_budget_candidates(
                    &global_budget_candidate_mask,
                    layer_index,
                );
                let remaining_secs = deadline
                    .map(|d| {
                        let now = Instant::now();
                        if now >= d {
                            0.0
                        } else {
                            d.duration_since(now).as_secs_f64()
                        }
                    })
                    .unwrap_or(0.0);
                debug!(
                    "CROWN-IBP DAG: node '{}' per-node budget {:.1}s < {:.1}s floor ({} tightening targets remain, {:.1}s left), using IBP",
                    node_name,
                    if remaining_global_candidates == 0 {
                        0.0
                    } else {
                        remaining_secs / remaining_global_candidates as f64
                    },
                    per_node_floor_secs,
                    remaining_global_candidates,
                    remaining_secs,
                );
                skip_count += 1;
                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                provenance.insert(
                    node_name.clone(),
                    BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                    ),
                );
                fallback_events.push(CrownIbpFallbackEvent {
                    layer_index,
                    layer_type,
                    reason: CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                    details: format!(
                        "node '{}' per-node budget below {:.1}s floor ({} tightening targets remain)",
                        node_name, per_node_floor_secs, remaining_global_candidates,
                    ),
                });
                continue;
            }

            // Patches-eligible targets use the collector-specific entry point
            // that overrides matrix conv_mode for this cut-free path (#3813).
            let node_start = Instant::now();
            let layer_type_for_log = layer_type.clone();

            // #margin-subset-seed: OUTPUT-node margin-subset tightening. When
            // the initial-bounds scope published the spec-referenced OUTPUT
            // indices AND this is the OUTPUT node at/above the engagement
            // width (see `margin_subset_indices`), seed ONLY the k referenced
            // identity rows (each bit-identical in semantics to its full-width
            // counterpart by row-independence) and SCATTER them over the
            // node's sound IBP bounds; the scattered map then flows through
            // the SAME shape-restore + IBP-intersection path as a full map,
            // so every row remains a valid enclosure. Engaged proactively:
            // the k-row backward is cheaper than full-width even when the
            // conv memory cap would not trip. On ANY error the existing
            // full-width path runs unchanged (fail-open).
            //
            // Cache note (#cgan-collection-cache): a collection stored while
            // engaged carries IBP-loose unreferenced OUTPUT rows to later
            // same-box lookups — always sound (every row is a valid
            // enclosure); the verdict path derives root bounds from the
            // objective backward, never from the collected OUTPUT rows.
            let margin_subset_bound = margin_subset_output_node
                .as_deref()
                .filter(|output_node| *output_node == node_name.as_str())
                .and_then(|_| crate::output_margin_seed::margin_subset_indices(ibp_bound.len()))
                .and_then(|indices| {
                    match self.propagate_crown_to_node_subset(
                        input,
                        node_name,
                        &crown_ibp_bounds,
                        &ibp_bounds,
                        engine,
                        "CROWN-IBP-margin-subset",
                        per_node_deadline,
                        is_patches_target,
                        &indices,
                        cut_ctx.as_ref(),
                    ) {
                        Ok((lower_rows, upper_rows)) => {
                            match scatter_margin_rows_over_bounds(
                                ibp_bound,
                                &indices,
                                &lower_rows,
                                &upper_rows,
                            ) {
                                Ok(bounds) => {
                                    info!(
                                        "CROWN-IBP DAG: output node '{}' margin-subset seed \
                                         engaged (k={} of {} rows; scattered over IBP)",
                                        node_name,
                                        indices.len(),
                                        ibp_bound.len()
                                    );
                                    Some(bounds)
                                }
                                Err(e) => {
                                    debug!(
                                        "CROWN-IBP DAG: output node '{}' margin-subset scatter \
                                         failed ({e}); falling back to full-width backward",
                                        node_name
                                    );
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            debug!(
                                "CROWN-IBP DAG: output node '{}' margin-subset backward failed \
                                 ({e}); falling back to full-width backward",
                                node_name
                            );
                            None
                        }
                    }
                });

            let crown_result = if let Some(bounds) = margin_subset_bound {
                Ok(bounds)
            } else if is_patches_target {
                self.propagate_crown_to_node_for_collector(
                    input,
                    node_name,
                    &crown_ibp_bounds,
                    &ibp_bounds,
                    engine,
                    per_node_deadline,
                    chunk_override,
                    cut_ctx.as_ref(),
                )
            } else {
                self.propagate_crown_to_node(
                    input,
                    node_name,
                    &crown_ibp_bounds,
                    &ibp_bounds,
                    engine,
                    per_node_deadline,
                    chunk_override,
                    cut_ctx.as_ref(),
                )
            };
            match crown_result {
                Ok(crown_bound) => {
                    let forward_contract =
                        GraphTargetShapeContract::from_bounds(node_name, ibp_bound);
                    let crown_bound = match forward_contract.reshape_for_forward_match(
                        crown_bound,
                        ibp_bound,
                        "CROWN-IBP forward-shape restore",
                    ) {
                        Ok(reshaped) => Some(reshaped),
                        Err(NyError::ShapeMismatch { expected, got }) => {
                            let reason = CrownIbpFallbackReason::ShapeMismatch;
                            debug!(
                                "CROWN-IBP DAG: {} shape mismatch IBP={:?} vs CROWN={:?}, using IBP",
                                node_name, expected, got
                            );
                            crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                            provenance.insert(
                                node_name.clone(),
                                BoundsProvenance::ForwardFallback(reason),
                            );
                            fallback_events.push(CrownIbpFallbackEvent {
                                layer_index,
                                layer_type: layer_type.clone(),
                                reason,
                                details: format!(
                                    "node '{}' crown shape {:?} does not match forward shape {:?}",
                                    node_name, got, expected
                                ),
                            });
                            None
                        }
                        Err(err) => return Err(err),
                    };
                    if let Some(crown_bound) = crown_bound {
                        match ibp_bound.intersection_per_element(&crown_bound) {
                            // Per-element intersection succeeded (#2935).
                            Some((tightened, disjoint)) => {
                                if disjoint > 0 {
                                    debug!(
                                        "CROWN-IBP DAG: {} per-element intersection: {} of {} elements disjoint, used union fallback",
                                        node_name, disjoint, tightened.len()
                                    );
                                }
                                crown_ibp_bounds.insert(node_name.clone(), tightened);
                                provenance.insert(node_name.clone(), BoundsProvenance::Crown);
                            }
                            // NaN or shape mismatch — full IBP fallback.
                            None => {
                                let reason = CrownIbpFallbackReason::EmptyIntersection;
                                debug!(
                                    "CROWN-IBP DAG: {} IBP/CROWN intersection failed (NaN), using IBP",
                                    node_name,
                                );
                                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                                provenance.insert(
                                    node_name.clone(),
                                    BoundsProvenance::ForwardFallback(reason),
                                );
                                fallback_events.push(CrownIbpFallbackEvent {
                                    layer_index,
                                    layer_type: layer_type.clone(),
                                    reason,
                                    details: format!(
                                        "node '{}' forward/CROWN intersection failed (NaN) for shape {:?}",
                                        node_name,
                                        ibp_bound.shape()
                                    ),
                                });
                            }
                        }
                    }
                }
                // #3166, #3602, #3499: UnsupportedOp/Configuration, ShapeMismatch,
                // or NumericalInstability from CROWN backward — IBP fallback is
                // sound. NumericalInstability catches non-finite pre-activation
                // bounds (e.g., from Sqrt, Softmax) that prevent CROWN relaxation.
                // UnsupportedConfiguration now also covers per-node deadline
                // exceeded (#3499), which returns from propagate_crown_to_node_core.
                // CpuMemoryExceeded is the Conv2d backward memory-cap backstop
                // (#conv-crown-oom): the dense coefficient buffer would exceed the
                // per-buffer cap, so this target degrades to sound IBP instead of
                // OOMing. IBP bounds are a valid over-approximation.
                Err(
                    e @ (NyError::UnsupportedOp(_)
                    | NyError::UnsupportedConfiguration(_)
                    | NyError::ShapeMismatch { .. }
                    | NyError::NumericalInstability(_)
                    | NyError::CpuMemoryExceeded { .. }
                    | NyError::DeadlineExceeded(_)),
                ) => {
                    // #3795: structural match on DeadlineExceeded replaces string matching
                    let reason = if e.is_deadline_exceeded() {
                        CrownIbpFallbackReason::PerNodeDeadlineExceeded
                    } else if e.is_cpu_memory_exceeded() {
                        CrownIbpFallbackReason::MemoryBudgetExceeded
                    } else {
                        CrownIbpFallbackReason::CrownPropagationError
                    };
                    debug!(
                        "CROWN-IBP DAG: {} CROWN backward failed ({}), using IBP",
                        node_name, e,
                    );
                    crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                    provenance.insert(node_name.clone(), BoundsProvenance::ForwardFallback(reason));
                    fallback_events.push(CrownIbpFallbackEvent {
                        layer_index,
                        layer_type,
                        reason,
                        details: format!(
                            "node '{}' CROWN backward failed ({e}), IBP fallback",
                            node_name
                        ),
                    });
                }
                Err(e) => return Err(e),
            }
            let node_secs = node_start.elapsed().as_secs_f64();
            patches_budget.record_elapsed(is_patches_target, node_secs);
            crown_node_count += 1;
            crown_total_secs += node_secs;
            if node_secs > 0.5 {
                info!(
                    "CROWN-IBP DAG: node {}/{} '{}' ({}) took {node_secs:.3}s",
                    layer_index, total_nodes, node_name, layer_type_for_log,
                );
            }
        }

        let collection_secs = collection_start.elapsed().as_secs_f64();
        if collection_secs > 0.1 {
            info!(
                "CROWN-IBP DAG collection: {collection_secs:.3}s total, \
                 {crown_node_count} crown nodes ({crown_total_secs:.3}s), \
                 {skip_count} skipped, {demand_skip_count} demand-skipped, {total_nodes} total",
            );
        }
        // #crown-cut-segment: one-line sweep summary whenever the gate is on.
        if let Some(ctx) = cut_ctx.as_ref() {
            info!(
                "CROWN-IBP DAG cut-segment sweep: NY_CROWN_CUT_SEGMENT={cut_segment}, \
                 {crown_node_count} nodes swept, {} cuts used, {collection_secs:.3}s wall",
                ctx.cuts_used(),
            );
        }
        // #conv-patches-collect diagnostic (default-OFF): dump per-node provenance
        // for the spatial (3D) conv-graph nodes so a metaroom/cifar100 probe can
        // see exactly which deep conv targets tightened (Crown) vs fell back (and
        // why). stderr println so it survives the vnncomp log filter.
        if std::env::var_os("NY_CONV_PATCHES_DEBUG").is_some_and(|v| v != "0" && !v.is_empty()) {
            for node_name in exec_order.iter() {
                let Some(b) = crown_ibp_bounds.get(node_name) else {
                    continue;
                };
                if b.shape().len() != 3 {
                    continue;
                }
                let prov = provenance.get(node_name);
                eprintln!(
                    "[conv-patches-dbg] node={node_name} shape={:?} numel={} width={:.4} prov={:?}",
                    b.shape(),
                    b.len(),
                    b.max_width(),
                    prov,
                );
            }
        }

        Ok(GraphCrownIbpBoundsResult {
            bounds: crown_ibp_bounds,
            provenance,
            fallback_events,
        })
    }
}

/// #margin-subset-seed: scatter k tight CROWN rows over the node's sound
/// IBP/forward bounds.
///
/// Referenced flat positions take the CROWN row values; every other position
/// keeps `base`'s (sound, merely looser) enclosure. The caller then intersects
/// the result with IBP exactly as it does for full-width CROWN maps, so every
/// row of the final bound remains a valid enclosure regardless of which source
/// it came from.
fn scatter_margin_rows_over_bounds(
    base: &BoundedTensor,
    indices: &[usize],
    lower_rows: &[f32],
    upper_rows: &[f32],
) -> Result<BoundedTensor> {
    if indices.len() != lower_rows.len() || indices.len() != upper_rows.len() {
        return Err(NyError::InvalidSpec(format!(
            "margin-subset scatter: {} indices but {}/{} rows",
            indices.len(),
            lower_rows.len(),
            upper_rows.len()
        )));
    }
    let flat = base.flatten();
    let mut lower = flat.lower().to_owned();
    let mut upper = flat.upper().to_owned();
    for ((&idx, &lo), &up) in indices.iter().zip(lower_rows).zip(upper_rows) {
        if idx >= lower.len() {
            return Err(NyError::InvalidSpec(format!(
                "margin-subset scatter: index {idx} out of range for {} elements",
                lower.len()
            )));
        }
        lower[[idx]] = lo;
        upper[[idx]] = up;
    }
    // Allow infinite endpoints (a degraded CROWN row is still a valid, merely
    // vacuous, enclosure); NaN is rejected downstream by the IBP intersection.
    let scattered = BoundedTensor::new_allow_infinite(lower, upper)?;
    scattered.reshape(base.shape())
}

#[cfg(test)]
mod margin_subset_collector_tests {
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use crate::output_margin_seed::MarginOutputSeedGuard;
    use crate::types::BoundsProvenance;
    use ndarray::{arr1, arr2, Array2};
    use ny_tensor::BoundedTensor;

    /// input(2) -> Linear(2->3) "pre" -> ReLU "act" -> Linear(3->600) "out".
    /// 600 outputs put the OUTPUT node at/above the margin-subset engagement
    /// width; the unstable ReLUs make CROWN strictly tighter than IBP.
    fn wide_output_net() -> (GraphNetwork, BoundedTensor) {
        let pre = LinearLayer::new(
            arr2(&[[1.0_f32, -0.5], [0.25, 0.75], [-0.6, 0.4]]),
            Some(arr1(&[0.05_f32, -0.1, 0.02])),
        )
        .expect("pre");
        // Deterministic mixed-sign weights so IBP loses correlations on
        // (essentially) every output row.
        let weights = Array2::from_shape_fn((600, 3), |(i, j)| {
            let v = ((i * 7 + j * 13) % 11) as f32 / 11.0 - 0.5;
            if v == 0.0 {
                0.3
            } else {
                v
            }
        });
        let out = LinearLayer::new(weights, None).expect("out");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("pre", Layer::Linear(pre)));
        graph.add_node(GraphNode::new(
            "act",
            Layer::ReLU(ReLULayer),
            vec!["pre".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(out),
            vec!["act".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5]).into_dyn(),
            arr1(&[1.0_f32, 0.75]).into_dyn(),
        )
        .expect("input");
        (graph, input)
    }

    /// #margin-subset-seed end-to-end through the collector: with published
    /// indices the OUTPUT node's referenced rows are BIT-IDENTICAL to the
    /// full-width collection's rows, every unreferenced row keeps the sound
    /// IBP enclosure, and provenance stays `Crown`. Without a publication the
    /// collection is byte-identical to the full-width behavior.
    #[test]
    fn collector_scatters_published_margin_rows_over_ibp() {
        let (graph, input) = wide_output_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP bounds");

        // Full-width reference collection (no publication on this thread).
        let full = graph
            .collect_crown_ibp_bounds_core_inner_with_cut_segment(
                &input,
                ibp.clone(),
                None,
                None,
                None,
                0,
            )
            .expect("full-width collection");

        // Published {5, 200}: the OUTPUT tighten seeds only those rows.
        let _guard = MarginOutputSeedGuard::publish(vec![200, 5]);
        let subset = graph
            .collect_crown_ibp_bounds_core_inner_with_cut_segment(
                &input,
                ibp.clone(),
                None,
                None,
                None,
                0,
            )
            .expect("margin-subset collection");

        let out_full = full.bounds.get("out").expect("full out");
        let out_subset = subset.bounds.get("out").expect("subset out");
        let out_ibp = ibp.get("out").expect("ibp out");
        assert_eq!(out_subset.shape(), out_full.shape());
        for i in 0..600 {
            if i == 5 || i == 200 {
                assert_eq!(
                    out_subset.lower()[[i]],
                    out_full.lower()[[i]],
                    "referenced lower row {i} must match the full-width collection"
                );
                assert_eq!(
                    out_subset.upper()[[i]],
                    out_full.upper()[[i]],
                    "referenced upper row {i} must match the full-width collection"
                );
            } else {
                assert_eq!(
                    out_subset.lower()[[i]],
                    out_ibp.lower()[[i]],
                    "unreferenced lower row {i} must keep the IBP enclosure"
                );
                assert_eq!(
                    out_subset.upper()[[i]],
                    out_ibp.upper()[[i]],
                    "unreferenced upper row {i} must keep the IBP enclosure"
                );
            }
        }
        assert!(matches!(
            subset.provenance.get("out"),
            Some(BoundsProvenance::Crown)
        ));
        // Meaningfulness guard: full-width CROWN actually tightens the
        // referenced rows past IBP (otherwise the equalities above are vacuous).
        assert!(
            [5_usize, 200].iter().any(|&i| {
                out_full.lower()[[i]] > out_ibp.lower()[[i]]
                    || out_full.upper()[[i]] < out_ibp.upper()[[i]]
            }),
            "CROWN must beat IBP on a referenced row"
        );
        // Non-output nodes are untouched by the publication.
        let pre_full = full.bounds.get("pre").expect("full pre");
        let pre_subset = subset.bounds.get("pre").expect("subset pre");
        assert_eq!(pre_subset.lower(), pre_full.lower());
        assert_eq!(pre_subset.upper(), pre_full.upper());
    }
}

#[cfg(test)]
mod margin_subset_scatter_tests {
    use super::scatter_margin_rows_over_bounds;
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    fn base_bounds() -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[6]), vec![-10.0f32; 6]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[6]), vec![10.0f32; 6]).unwrap(),
        )
        .unwrap()
    }

    /// Referenced rows take the CROWN values, unreferenced rows keep the base
    /// enclosure, and the (already-tighter) intersection with the base is a
    /// sound enclosure row-for-row.
    #[test]
    fn scatter_places_rows_and_intersection_stays_sound() {
        let base = base_bounds();
        let scattered =
            scatter_margin_rows_over_bounds(&base, &[1, 4], &[-1.5, 2.0], &[0.5, 3.25]).unwrap();
        assert_eq!(scattered.shape(), base.shape());
        let lo = scattered.lower();
        let up = scattered.upper();
        for i in 0..6 {
            match i {
                1 => assert_eq!((lo[[i]], up[[i]]), (-1.5, 0.5)),
                4 => assert_eq!((lo[[i]], up[[i]]), (2.0, 3.25)),
                _ => assert_eq!((lo[[i]], up[[i]]), (-10.0, 10.0)),
            }
        }
        // The collector's IBP intersection keeps every row inside the sound
        // IBP enclosure.
        let (tightened, disjoint) = base
            .intersection_per_element(&scattered)
            .expect("intersection succeeds");
        assert_eq!(disjoint, 0);
        for i in 0..6 {
            assert!(tightened.lower()[[i]] >= base.lower()[[i]]);
            assert!(tightened.upper()[[i]] <= base.upper()[[i]]);
            assert!(tightened.lower()[[i]] <= tightened.upper()[[i]]);
        }
    }

    /// Multi-dimensional base: the scatter runs over the FLAT index space and
    /// restores the base's shape.
    #[test]
    fn scatter_restores_multi_dim_shape() {
        let base = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0f32; 6]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32; 6]).unwrap(),
        )
        .unwrap();
        let scattered = scatter_margin_rows_over_bounds(&base, &[5], &[0.25], &[0.75]).unwrap();
        assert_eq!(scattered.shape(), &[2, 3]);
        assert_eq!(scattered.lower()[[1, 2]], 0.25);
        assert_eq!(scattered.upper()[[1, 2]], 0.75);
        assert_eq!(scattered.lower()[[0, 0]], 0.0);
    }

    /// Malformed requests fail (the consume site falls back to full-width).
    #[test]
    fn scatter_rejects_len_mismatch_and_out_of_range() {
        let base = base_bounds();
        assert!(scatter_margin_rows_over_bounds(&base, &[1, 4], &[0.0], &[0.0, 0.0]).is_err());
        assert!(scatter_margin_rows_over_bounds(&base, &[6], &[0.0], &[0.0]).is_err());
    }
}
