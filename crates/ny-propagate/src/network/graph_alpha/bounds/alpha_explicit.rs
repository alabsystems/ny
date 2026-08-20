// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit-alpha backward helpers for DAG α-CROWN.

use super::budget_policy::ObjectiveChunkSchedulingPlan;
use super::target_backward::{ObjectiveChunkFixedWavePlan, ObjectiveChunkRoutePlan};
use super::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(test)]
thread_local! {
    static ALPHA_INTERMEDIATE_COLLECTION_ENTRIES: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only observation scope for full explicit-alpha intermediate walks.
///
/// The typed cGAN transaction returns an already-complete sound reference map;
/// its dispatcher must not widen that one-target authority into this all-node
/// collection after the optimizer returns.
#[cfg(test)]
pub(crate) struct AlphaIntermediateCollectionEntryCounter {
    previous: Option<usize>,
}

#[cfg(test)]
impl AlphaIntermediateCollectionEntryCounter {
    pub(crate) fn start() -> Self {
        let previous = ALPHA_INTERMEDIATE_COLLECTION_ENTRIES.with(|slot| slot.replace(Some(0)));
        Self { previous }
    }

    pub(crate) fn entries(&self) -> usize {
        ALPHA_INTERMEDIATE_COLLECTION_ENTRIES.with(|slot| {
            slot.get()
                .expect("alpha intermediate collection counter scope must still be active")
        })
    }
}

#[cfg(test)]
impl Drop for AlphaIntermediateCollectionEntryCounter {
    fn drop(&mut self) {
        ALPHA_INTERMEDIATE_COLLECTION_ENTRIES.with(|slot| slot.set(self.previous));
    }
}

#[cfg(test)]
fn record_alpha_intermediate_collection_entry() {
    ALPHA_INTERMEDIATE_COLLECTION_ENTRIES.with(|slot| {
        if let Some(entries) = slot.get() {
            slot.set(Some(entries.saturating_add(1)));
        }
    });
}

#[derive(Clone, Copy, Debug)]
struct ChunkAwareAlphaRoute {
    execution: ObjectiveChunkRoutePlan,
    scheduling: Option<ObjectiveChunkSchedulingPlan>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::network::graph_alpha) enum M1AlphaBudgetOutcome {
    NotAdmitted,
    BelowFloor,
    Allocate,
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::network::graph_alpha) enum M1AlphaTraceEvent {
    BudgetAdmission {
        node: String,
        outcome: M1AlphaBudgetOutcome,
        deadline_present: bool,
    },
    BackwardDispatch {
        node: String,
        retained_fixed_wave: bool,
    },
}

#[cfg(test)]
std::thread_local! {
    static M1_ALPHA_TRACE: std::cell::RefCell<Option<Vec<M1AlphaTraceEvent>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn record_m1_alpha_trace(event: M1AlphaTraceEvent) {
    M1_ALPHA_TRACE.with(|trace| {
        if let Some(events) = trace.borrow_mut().as_mut() {
            events.push(event);
        }
    });
}

#[cfg(test)]
pub(in crate::network::graph_alpha) fn run_with_m1_alpha_trace<T>(
    f: impl FnOnce() -> T,
) -> (T, Vec<M1AlphaTraceEvent>) {
    struct RestoreTrace(Option<Vec<M1AlphaTraceEvent>>);

    impl Drop for RestoreTrace {
        fn drop(&mut self) {
            M1_ALPHA_TRACE.with(|trace| {
                trace.replace(self.0.take());
            });
        }
    }

    let previous = M1_ALPHA_TRACE.with(|trace| trace.replace(Some(Vec::new())));
    let restore = RestoreTrace(previous);
    let output = f();
    let events = M1_ALPHA_TRACE.with(|trace| trace.replace(None).unwrap_or_default());
    drop(restore);
    (output, events)
}

impl GraphNetwork {
    /// Compute CROWN bounds for all nodes using explicit alpha values.
    ///
    /// `deadline` (#cifar100-alpha-interm) bounds the O(L²) per-node CROWN
    /// backward sweep: the deadline is checked between per-node computations,
    /// and on expiry the remaining (not-yet-tightened) nodes fall back to the
    /// sound IBP reference bounds. This prevents a deep ResNet's intermediate
    /// collection from overrunning the verifier wall-clock budget. Returning
    /// IBP bounds for the un-collected tail is sound — they are valid bounds,
    /// just looser than the alpha-tightened CROWN bounds.
    pub(in crate::network::graph_alpha) fn collect_crown_bounds_with_alpha(
        &self,
        input: &BoundedTensor,
        ibp_bounds: &HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        #[cfg(test)]
        record_alpha_intermediate_collection_entry();

        let exec_order = self.exec_order()?;
        let mut crown_bounds: HashMap<String, BoundedTensor> = HashMap::new();

        // For each node, run backward CROWN with alpha values.
        for node_name in exec_order {
            // Deadline check (#cifar100-alpha-interm): bail before computing the
            // next per-node CROWN backward if the budget is exhausted. Remaining
            // nodes use IBP reference bounds (sound, looser) so the verifier
            // does not overrun its wall-clock budget on O(L²) collection.
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    info!(
                        "α-CROWN: deadline exceeded during intermediate collection at node '{}', \
                         filling remaining nodes from IBP reference bounds",
                        node_name
                    );
                    for remaining in exec_order {
                        if !crown_bounds.contains_key(remaining) {
                            if let Some(ibp) = ibp_bounds.get(remaining) {
                                crown_bounds.insert(remaining.clone(), ibp.clone());
                            }
                        }
                    }
                    return Ok(crown_bounds);
                }
            }

            match self.propagate_crown_to_node_with_alpha(
                input,
                node_name,
                &crown_bounds,
                ibp_bounds,
                alpha_state,
                engine,
                deadline,
            ) {
                Ok(bounds) => {
                    // Intersect this node's CROWN backward bound with the always-available
                    // IBP bound BEFORE caching, so every downstream relaxation is built on
                    // the TIGHTER intermediate pre-activation (not the looser raw CROWN that
                    // was only IBP-intersected at the end of collection). SOUND: both enclose
                    // the node's true reachable pre-activation set, so the per-element
                    // intersection [max(l),min(u)] (union on disjoint) still encloses it;
                    // None (NaN/shape mismatch) keeps the CROWN bound (a sound enclosure).
                    // Same proven pattern as the per-iteration output-bound intersection.
                    let cached = match ibp_bounds.get(node_name) {
                        Some(ibp) if ibp.shape() == bounds.shape() => bounds
                            .intersection_per_element(ibp)
                            .map(|(t, _)| t)
                            .unwrap_or(bounds),
                        _ => bounds,
                    };
                    crown_bounds.insert(node_name.clone(), cached);
                }
                // Expected fallback: unsupported op or shape mismatch means this
                // node can't use CROWN backward pass, so IBP is the correct fallback.
                // #3166: Catch UnsupportedConfiguration alongside UnsupportedOp.
                // #3602: Catch ShapeMismatch from complex DAG topologies (e.g., Concat
                // of heterogeneous sub-graphs) where CROWN backward produces intermediate
                // tensors with mismatched dimensions. IBP handles these correctly.
                // CpuMemoryExceeded: Conv2d backward memory-cap backstop
                // (#conv-crown-oom). IBP is the sound fallback.
                Err(e)
                    if matches!(
                        e,
                        NyError::UnsupportedOp(_)
                            | NyError::UnsupportedConfiguration(_)
                            | NyError::ShapeMismatch { .. }
                            | NyError::CpuMemoryExceeded { .. }
                            | NyError::DeadlineExceeded(_)
                    ) =>
                {
                    // Name the ACTUAL error variant (#cgan-alpha-chunk): the old
                    // generic "unsupported/shape mismatch" message conflated a
                    // genuine UnsupportedOp with the memory-budget (CpuMemoryExceeded)
                    // and per-node deadline (DeadlineExceeded) backstops, hiding which
                    // fired — the memory case is now rerouted through chunking, so a
                    // persistent fallback here is a deadline or true-unsupported one.
                    warn!(
                        "α-CROWN: node '{}' CROWN backward fell back to IBP: {}",
                        node_name, e
                    );
                    if let Some(ibp) = ibp_bounds.get(node_name) {
                        crown_bounds.insert(node_name.clone(), ibp.clone());
                    }
                }
                // #3107: LayerError may wrap critical errors — inspect source before fallback.
                Err(NyError::LayerError { source, .. })
                    if matches!(
                        source.as_ref(),
                        NyError::SoundnessRefusal(_)
                            | NyError::NumericalInstability(_)
                            | NyError::InternalError(_)
                    ) =>
                {
                    return Err(*source);
                }
                Err(NyError::LayerError { .. }) => {
                    warn!(
                        "α-CROWN: node '{}' CROWN backward unsupported (wrapped), falling back to IBP",
                        node_name
                    );
                    if let Some(ibp) = ibp_bounds.get(node_name) {
                        crown_bounds.insert(node_name.clone(), ibp.clone());
                    }
                }
                // Unexpected errors (internal error, soundness refusal, etc.)
                // should not be silently swallowed — propagate them (#2032, #1941).
                Err(err) => return Err(err),
            }
        }

        // #conv-patches-collect(alpha) diagnostic (default-OFF): dump per-node
        // width + provenance for the spatial (3D) conv-graph nodes on the ALPHA
        // intermediate path, so a metaroom probe can compare the alpha-tightened
        // deep-conv widths against the plain CROWN-IBP dump. Print-only; never
        // feeds a verdict.
        if std::env::var_os("NY_CONV_PATCHES_DEBUG").is_some_and(|v| v != "0" && !v.is_empty()) {
            for node_name in self.exec_order()? {
                let Some(b) = crown_bounds.get(node_name) else {
                    continue;
                };
                if b.shape().len() != 3 {
                    continue;
                }
                let ibp_w = ibp_bounds.get(node_name).map(|i| i.max_width());
                eprintln!(
                    "[conv-patches-alpha-dbg] node={node_name} shape={:?} numel={} \
                     alpha_width={:.4} ibp_width={:?}",
                    b.shape(),
                    b.len(),
                    b.max_width(),
                    ibp_w,
                );
            }
        }

        Ok(crown_bounds)
    }

    /// Compute CROWN bounds for a selected list of graph nodes using explicit
    /// alpha values, while falling back to the caller-supplied reference map
    /// for unsupported nodes.
    ///
    /// `deadline` (#cifar100-alpha-interm) bounds the O(L²) per-node CROWN
    /// backward sweep this performs to satisfy the requested `targets`. The
    /// deadline is checked between per-node computations; on expiry any
    /// not-yet-collected target falls back to its sound `reference_bounds`
    /// entry (valid, looser) so per-iteration reference refresh on a deep
    /// ResNet cannot overrun the verifier wall-clock budget.
    pub(in crate::network::graph_alpha) fn collect_selected_crown_bounds_with_alpha(
        &self,
        input: &BoundedTensor,
        targets: &[String],
        reference_bounds: &HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        self.collect_selected_crown_bounds_with_alpha_mode(
            input,
            targets,
            reference_bounds,
            alpha_state,
            engine,
            deadline,
            false,
        )
    }

    /// `targets_only = true` (#image-node-crown): compute CROWN walks ONLY for the
    /// target nodes, using the reference map as the relaxation source for every
    /// other node instead of building the full exec-order crown cache. Measured
    /// on cgan: the cache sweep recomputes the heavy generator prefix
    /// (ConvT_4/7/10 ≈ 170s) before reaching downstream targets, so a bounded
    /// outlier pass envelope-bails and reference-fills its targets without ever
    /// walking them. For the outlier pass the prefix recompute is pure waste —
    /// the reference map is already tight there (and the iterated passes
    /// progressively tighten it where it is not). `false` = byte-identical
    /// historical behavior for the alpha-refresh path.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::network::graph_alpha) fn collect_selected_crown_bounds_with_alpha_mode(
        &self,
        input: &BoundedTensor,
        targets: &[String],
        reference_bounds: &HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
        targets_only: bool,
    ) -> Result<HashMap<String, BoundedTensor>> {
        if targets.is_empty() {
            return Ok(HashMap::new());
        }

        let exec_order = self.exec_order()?;
        let exec_node_set: std::collections::HashSet<String> = exec_order.iter().cloned().collect();
        let target_set: std::collections::HashSet<String> = targets.iter().cloned().collect();
        let mut crown_bounds: HashMap<String, BoundedTensor> = HashMap::new();
        let mut selected: HashMap<String, BoundedTensor> = HashMap::with_capacity(targets.len());

        let chunk_aware_budget = budget_policy::crown_chunk_aware_budget_enabled();
        // Disabled mode deliberately does not enter this closure. It retains
        // the historical lazy `alpha_target_chunk_override` calls in both the
        // gate and each backward. Armed M1 resolves TARGET routes once so the
        // denominator, numerator, and execution consume one central plan.
        let chunk_aware_target_routes: Option<HashMap<String, Option<ChunkAwareAlphaRoute>>> =
            chunk_aware_budget.then(|| {
                let dense_budget = cpu_crown_dense_budget_bytes();
                targets
                    .iter()
                    .filter_map(|target| {
                        reference_bounds.get(target).map(|bounds| {
                            let route = budget_policy::auto_objective_chunk_route_plan(
                                self,
                                target,
                                bounds,
                                input.len(),
                                dense_budget,
                                deadline.is_some(),
                                true,
                            )
                            .map(|execution| ChunkAwareAlphaRoute {
                                execution,
                                scheduling: budget_policy::objective_chunk_scheduling_plan(
                                    bounds.len(),
                                    execution,
                                    deadline.is_some(),
                                    false,
                                ),
                            });
                            (target.clone(), route)
                        })
                    })
                    .collect()
            });

        // STEP 2b (#cgan-alpha-refresh-budget): cgan-like graphs give each node its
        // own equal-share time window (capped at the preset per-node cap) within the
        // overall refresh envelope, instead of racing one flat shared deadline that
        // the first expensive generator target would consume — starving the rest
        // into a reference fallback. Non-cgan graphs keep the flat `deadline`.
        let use_per_node_budget = match chunk_aware_target_routes.as_ref() {
            Some(routes) => self.alpha_refresh_uses_per_node_budget_with_plans(targets, routes),
            None => self.alpha_refresh_uses_per_node_budget(targets, reference_bounds),
        };

        for node_name in exec_order {
            if selected.len() == targets.len() {
                break;
            }
            if targets_only && !target_set.contains(node_name.as_str()) {
                continue; // #image-node-crown: reference relaxations suffice
            }

            // Deadline check (#cifar100-alpha-interm): bail before the next
            // per-node CROWN backward if the OVERALL envelope is exhausted, filling
            // any outstanding targets from the sound reference bounds. (Per-node
            // windows, below, only ever end EARLIER than the envelope, so a single
            // slow target falls back without triggering this whole-sweep bail.)
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    info!(
                        "α-CROWN: deadline exceeded during selected intermediate collection at \
                         node '{}', filling outstanding targets from reference bounds",
                        node_name
                    );
                    for target in targets {
                        if !selected.contains_key(target) {
                            let fallback =
                                reference_bounds.get(target).cloned().ok_or_else(|| {
                                    NyError::InvalidSpec(format!(
                                        "Reference bounds for selected node '{}' not found",
                                        target
                                    ))
                                })?;
                            selected.insert(target.clone(), fallback);
                        }
                    }
                    return Ok(selected);
                }
            }

            // STEP 2b: the per-node window. Dark M1 preserves the parent policy
            // exactly. Armed M1 admits only still-unselected requested targets:
            // numerator and denominator are therefore the SAME set. A non-target
            // or below-floor target takes its reference bound directly; neither
            // can reinterpret "no allocation" as the full remaining envelope.
            let chunk_route = chunk_aware_target_routes
                .as_ref()
                .and_then(|routes| routes.get(node_name))
                .copied()
                .flatten();
            let mut armed_budget_reference_fallback = false;
            let node_deadline = if use_per_node_budget {
                match deadline {
                    Some(env) => {
                        let now = Instant::now();
                        let remaining = env.saturating_duration_since(now).as_secs_f64();
                        // #cgan-collection-cost-weight (refresh port): COST-PROPORTIONAL
                        // windows with the dim-scaled cap — the policy the CROWN-IBP
                        // collector validated (crown_tighten.rs sum_remaining_budget_weights
                        // + compute_weighted_per_node_budget_secs). The former EQUAL share
                        // made the 28,800-dim generator target structurally unreachable on
                        // cgan (its ~95-112s refresh needs remaining >= ~8x95s under equal
                        // share), so its α-tightened bound was never installed and the map
                        // kept the loose ±2500 reference enclosure — the measured cgan
                        // verdict bottleneck. Armed M1 models only the invariant
                        // fixed-wave route; sequential/adaptive routes retain raw
                        // rows.
                        let raw_dims_of = |name: &str| -> usize {
                            reference_bounds.get(name).map_or(0, BoundedTensor::len)
                        };
                        let work_weight_of = |name: &str| -> f64 {
                            let route = chunk_aware_target_routes
                                .as_ref()
                                .and_then(|routes| routes.get(name))
                                .copied()
                                .flatten()
                                .and_then(|route| route.scheduling);
                            budget_policy::demanded_target_work_weight(
                                raw_dims_of(name),
                                route,
                                true,
                                chunk_aware_budget,
                            )
                        };
                        if chunk_aware_budget {
                            let mut seen = std::collections::HashSet::new();
                            let admitted_targets: Vec<&str> = targets
                                .iter()
                                .map(String::as_str)
                                .filter(|target| seen.insert(*target))
                                .filter(|target| !selected.contains_key(*target))
                                .filter(|target| reference_bounds.contains_key(*target))
                                .filter(|target| exec_node_set.contains(*target))
                                .collect();
                            let admitted = admitted_targets.contains(&node_name.as_str());
                            let admitted_weight_sum: f64 = admitted_targets
                                .iter()
                                .map(|target| work_weight_of(target))
                                .filter(|weight| weight.is_finite() && *weight > 0.0)
                                .sum();
                            let this_weight = if admitted {
                                work_weight_of(node_name)
                            } else {
                                0.0
                            };
                            let cap_dims = budget_policy::weighted_budget_cap_dims(
                                this_weight,
                                raw_dims_of(node_name) as f64,
                                true,
                            );
                            let admission = budget_policy::admitted_weighted_budget_secs(
                                admitted,
                                remaining,
                                admitted_weight_sum,
                                this_weight,
                                cap_dims,
                                &self.crown_ibp_per_node_time_budget,
                            );
                            #[cfg(test)]
                            record_m1_alpha_trace(M1AlphaTraceEvent::BudgetAdmission {
                                node: node_name.clone(),
                                outcome: match admission {
                                    budget_policy::WeightedBudgetAdmission::NotAdmitted => {
                                        M1AlphaBudgetOutcome::NotAdmitted
                                    }
                                    budget_policy::WeightedBudgetAdmission::BelowFloor => {
                                        M1AlphaBudgetOutcome::BelowFloor
                                    }
                                    budget_policy::WeightedBudgetAdmission::Allocate(_) => {
                                        M1AlphaBudgetOutcome::Allocate
                                    }
                                },
                                deadline_present: matches!(
                                    admission,
                                    budget_policy::WeightedBudgetAdmission::Allocate(_)
                                ),
                            });
                            match admission {
                                budget_policy::WeightedBudgetAdmission::Allocate(secs) => {
                                    Some((now + Duration::from_secs_f64(secs)).min(env))
                                }
                                budget_policy::WeightedBudgetAdmission::NotAdmitted
                                | budget_policy::WeightedBudgetAdmission::BelowFloor => {
                                    armed_budget_reference_fallback = true;
                                    None
                                }
                            }
                        } else {
                            // Historical default-dark allocation, including its
                            // lazy raw-row numerator and None=>envelope fallback.
                            let remaining_weight_sum: f64 = targets
                                .iter()
                                .filter(|target| !selected.contains_key(target.as_str()))
                                .map(|target| work_weight_of(target))
                                .filter(|weight| weight.is_finite() && *weight > 0.0)
                                .sum();
                            let this_weight = work_weight_of(node_name);
                            let cap_dims = budget_policy::weighted_budget_cap_dims(
                                this_weight,
                                raw_dims_of(node_name) as f64,
                                false,
                            );
                            match budget_policy::compute_weighted_per_node_budget_secs(
                                remaining,
                                remaining_weight_sum,
                                this_weight,
                                cap_dims,
                                &self.crown_ibp_per_node_time_budget,
                            ) {
                                Some(secs) => Some((now + Duration::from_secs_f64(secs)).min(env)),
                                None => Some(env),
                            }
                        }
                    }
                    None => None,
                }
            } else {
                deadline
            };

            let backward = if armed_budget_reference_fallback {
                reference_bounds.get(node_name).cloned().ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Reference bounds for budget-fallback node '{}' not found",
                        node_name
                    ))
                })
            } else if chunk_aware_budget && target_set.contains(node_name.as_str()) {
                let retained_fixed_waves = chunk_route
                    .and_then(|route| route.scheduling)
                    .map(|plan| plan.fixed_waves);
                #[cfg(test)]
                record_m1_alpha_trace(M1AlphaTraceEvent::BackwardDispatch {
                    node: node_name.clone(),
                    retained_fixed_wave: retained_fixed_waves.is_some(),
                });
                self.propagate_crown_to_node_with_alpha_and_chunk_override(
                    input,
                    node_name,
                    &crown_bounds,
                    reference_bounds,
                    alpha_state,
                    engine,
                    node_deadline,
                    chunk_route.map(|route| route.execution.requested_rows),
                    retained_fixed_waves,
                )
            } else {
                // Historical lazy route resolution for disabled M1 and for
                // non-denominator nodes in an armed targets-only=false sweep.
                #[cfg(test)]
                record_m1_alpha_trace(M1AlphaTraceEvent::BackwardDispatch {
                    node: node_name.clone(),
                    retained_fixed_wave: false,
                });
                self.propagate_crown_to_node_with_alpha(
                    input,
                    node_name,
                    &crown_bounds,
                    reference_bounds,
                    alpha_state,
                    engine,
                    node_deadline,
                )
            };
            let bounds = match backward {
                Ok(bounds) => bounds,
                // CpuMemoryExceeded: Conv2d backward memory-cap backstop
                // (#conv-crown-oom). Reference (IBP) bounds are the sound fallback.
                // #cgan-alpha-chunk: name the ACTUAL error — the memory case now
                // reroutes through chunking, so a persistent fallback here is a
                // deadline (mid-size targets under the refresh slice) or a genuine
                // unsupported op, not the memory abort the old message implied.
                Err(e)
                    if matches!(
                        e,
                        NyError::UnsupportedOp(_)
                            | NyError::UnsupportedConfiguration(_)
                            | NyError::ShapeMismatch { .. }
                            | NyError::CpuMemoryExceeded { .. }
                            | NyError::DeadlineExceeded(_)
                    ) =>
                {
                    warn!(
                        "α-CROWN: selected node '{}' CROWN backward fell back to reference bounds: {}",
                        node_name, e
                    );
                    reference_bounds.get(node_name).cloned().ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "Reference bounds for selected node '{}' not found",
                            node_name
                        ))
                    })?
                }
                Err(NyError::LayerError { source, .. })
                    if matches!(
                        source.as_ref(),
                        NyError::SoundnessRefusal(_)
                            | NyError::NumericalInstability(_)
                            | NyError::InternalError(_)
                    ) =>
                {
                    return Err(*source);
                }
                Err(NyError::LayerError { .. }) => {
                    warn!(
                        "α-CROWN: selected node '{}' CROWN backward unsupported (wrapped), falling back to reference bounds",
                        node_name
                    );
                    reference_bounds.get(node_name).cloned().ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "Reference bounds for selected node '{}' not found",
                            node_name
                        ))
                    })?
                }
                Err(err) => return Err(err),
            };

            // Intersect with the always-available IBP/reference bound before caching, so
            // downstream relaxations AND the returned selected bound use the tighter
            // intermediate pre-activation. SOUND: both enclose the node's reachable set, so
            // the per-element intersection (union on disjoint) still encloses it; None keeps
            // the CROWN bound. No-op when `bounds` already fell back to the reference bound.
            let bounds = match reference_bounds.get(node_name) {
                Some(ibp) if ibp.shape() == bounds.shape() => bounds
                    .intersection_per_element(ibp)
                    .map(|(t, _)| t)
                    .unwrap_or(bounds),
                _ => bounds,
            };
            crown_bounds.insert(node_name.clone(), bounds.clone());
            if target_set.contains(node_name) {
                selected.insert(node_name.clone(), bounds);
            }
        }

        if selected.len() != targets.len() {
            let missing: Vec<_> = targets
                .iter()
                .filter(|target| !selected.contains_key(*target))
                .cloned()
                .collect();
            return Err(NyError::InvalidSpec(format!(
                "Selected α-CROWN bounds missing target nodes {:?}",
                missing
            )));
        }

        Ok(selected)
    }

    /// Run backward CROWN from a target node using explicit alpha values.
    ///
    /// Uses patches override: α-CROWN produces concretized scalar bounds and
    /// per-neuron alpha parameters — neither depends on whether the CROWN
    /// backward used patches or dense representation. Only BaB CROWN with
    /// cutting planes needs dense A-matrices (#3813).
    ///
    /// `deadline` (#w4-refresh-deadline) is threaded into the per-target backward
    /// loop as the per-node deadline: a single wide-target spec-batch backward on
    /// a deep conv resnet was measured at ~26s UN-interruptible (the per-iteration
    /// reference refresh overran the verifier budget by that residual). The loop
    /// checks it between node steps and returns `DeadlineExceeded`, which every
    /// caller already catches with a sound (reference/IBP) fallback.
    /// Over-budget objective-chunk size for an α-CROWN intermediate target
    /// (#cgan-alpha-chunk). Mirrors the CROWN-IBP collector's gate verbatim
    /// (crown_tighten.rs:196-210): a target whose dense identity pair exceeds the
    /// CPU CROWN budget AND cannot start in patches reroutes through the
    /// bound-equivalent objective-chunked backward (`propagate_crown_to_node_chunked`)
    /// instead of degrading to the looser IBP/reference bound — the exact fallback
    /// that made cGAN generator targets (ConvTranspose_10 / BatchNormalization_11,
    /// 28,800-dim) surface `CpuMemoryExceeded` and drop to IBP. `None` (under
    /// budget) keeps the single-pass path byte-for-byte. Peak memory is bounded by
    /// the chunk; a chunked node that is still too slow degrades to IBP via the
    /// per-node deadline — sound either way (both enclose the reachable set).
    ///
    /// #patches-dense-peak — THE REROUTE IS CURRENTLY UNREACHABLE FROM THIS LANE
    /// WHENEVER A DEADLINE IS PRESENT. Recorded here so the next attempt does not
    /// re-land the predicate widening that was already tried and reverted.
    ///
    /// `cifar_bias_field_46` / `/layers.4/Relu` degrades with
    /// `CpuMemoryExceeded { site: "patches full dense materialization",
    /// required: 6_445_080_584, budget: 6_442_450_944 }`, caught by this
    /// function's caller at the `Err(e) if matches!(.., CpuMemoryExceeded)` arm
    /// of `collect_selected_crown_bounds_with_alpha_mode` and answered with the
    /// reference bound. Two independent gates must fall for the objective-chunk
    /// reroute to replace that fallback, and only the FIRST is in this crate's
    /// budget policy:
    ///
    /// 1. `graph_native_target_exceeds_budget` charges the `[dim x dim]`
    ///    identity pair (2 GiB here) while the failing site charges six
    ///    `[rows x in_dim]` matrices (6 GiB here) — a 3x under-charge that is
    ///    false regardless of the `crown_ibp_target_can_start_in_patches`
    ///    exemption. See the `#patches-dense-peak` note on that predicate for
    ///    the exact decomposition and the honest cost model.
    ///
    /// 2. Even with `Some(C)` returned here, the chunk request only selects the
    ///    driver — it does not run it. `propagate_crown_to_node_core` hard-wires
    ///    `deadline_is_hard = per_node_deadline.is_some()`
    ///    (target_backward.rs), and `propagate_crown_to_node_chunked` opens with
    ///    an unconditional `if deadline_is_hard { return Err(
    ///    UnsupportedConfiguration("finite objective-chunk target backward is
    ///    not cooperatively bounded ..")) }`. Every scored run carries a
    ///    deadline, so on this benchmark the reroute is refused at driver entry
    ///    and `UnsupportedConfiguration` lands on the SAME reference-bound arm
    ///    the memory error did. The target degrades either way; only the log
    ///    line changes.
    ///
    /// That second guard is the same PRESENCE-not-EXPIRY shape the
    /// `NY_PATCHES_FINITE_EXPIRY` family repairs, and the crate already owns the
    /// helper it should use —
    /// `network::core::sequential::crown::patches_step::hard_finite_authority_refuses_patches(deadline_is_hard, deadline)`,
    /// which returns `true` only once the deadline has actually EXPIRED when the
    /// authority is armed. The chunk driver already re-checks expiry at entry
    /// and polls between chunks, so deciding this refusal by expiry costs no
    /// interruptibility. Until that one call site changes, widening the budget
    /// predicate here buys nothing, so it is intentionally left alone.
    fn alpha_target_chunk_override(
        &self,
        node_name: &str,
        input_dim: usize,
        ref_bounds: &HashMap<String, BoundedTensor>,
    ) -> Option<usize> {
        let budget = cpu_crown_dense_budget_bytes();
        let ibp = ref_bounds.get(node_name)?;
        if self.graph_native_target_exceeds_budget(node_name, ibp, budget) {
            Some(budget_policy::auto_objective_chunk_rows(
                ibp.len(),
                input_dim,
                budget,
            ))
        } else {
            None
        }
    }

    /// STEP 2b gate (#cgan-alpha-refresh-budget): does this graph's per-iteration
    /// α refresh need PER-NODE time windows instead of the single flat
    /// allowance left in the cumulative refresh pool? True only for cgan-like
    /// generators — a ConvTranspose layer is present AND at least one refresh
    /// target is over-budget (trips the objective-chunk reroute, so its chunked
    /// f64 backward cannot fit an equal split of the shared slice). For every
    /// other graph (e.g. cifar100 resnet: Conv2d only, targets under budget) this
    /// is false, so the refresh keeps the flat shared deadline byte-for-byte and
    /// the BaB budget the cumulative `0.25` pool protects is untouched.
    fn alpha_refresh_uses_per_node_budget(
        &self,
        targets: &[String],
        reference_bounds: &HashMap<String, BoundedTensor>,
    ) -> bool {
        let has_convtranspose = self.nodes.values().any(|n| {
            matches!(
                n.layer,
                Layer::ConvTranspose2d(_) | Layer::ConvTranspose1d(_)
            )
        });
        has_convtranspose
            && targets.iter().any(|t| {
                // Only PRESENCE is read here, and presence turns on
                // `graph_native_target_exceeds_budget` alone — the row count is
                // discarded — so the input dimension cannot change this answer.
                self.alpha_target_chunk_override(t, 0, reference_bounds)
                    .is_some()
            })
    }

    /// Armed-M1 equivalent of [`Self::alpha_refresh_uses_per_node_budget`].
    ///
    /// The caller has already resolved target-only execution plans, so this
    /// consumes those retained plans instead of rediscovering routes.
    fn alpha_refresh_uses_per_node_budget_with_plans(
        &self,
        targets: &[String],
        plans: &HashMap<String, Option<ChunkAwareAlphaRoute>>,
    ) -> bool {
        let has_convtranspose = self.nodes.values().any(|n| {
            matches!(
                n.layer,
                Layer::ConvTranspose2d(_) | Layer::ConvTranspose1d(_)
            )
        });
        has_convtranspose
            && targets
                .iter()
                .any(|target| plans.get(target).copied().flatten().is_some())
    }

    pub(in crate::network::graph_alpha) fn propagate_crown_to_node_with_alpha(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &HashMap<String, BoundedTensor>,
        ibp_bounds: &HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        // #cgan-alpha-chunk: derive the objective-chunk override HERE (from the
        // reference map every caller already passes) rather than threading it
        // through all call sites. Over-budget generator targets (ConvTranspose/BN)
        // currently fail with CpuMemoryExceeded in EVERY caller and degrade to IBP;
        // rerouting them to the bound-equivalent chunked backward is strictly an
        // improvement everywhere, and under-budget targets stay byte-identical
        // (`None` → the existing single-pass path).
        let chunk_override = self.alpha_target_chunk_override(target_node, input.len(), ibp_bounds);
        self.propagate_crown_to_node_with_alpha_and_chunk_override(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            engine,
            deadline,
            chunk_override,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn propagate_crown_to_node_with_alpha_and_chunk_override(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &HashMap<String, BoundedTensor>,
        ibp_bounds: &HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
        chunk_override: Option<usize>,
        expected_fixed_waves: Option<ObjectiveChunkFixedWavePlan>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_to_node_core(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            Some(alpha_state),
            engine,
            "α-CROWN",
            deadline,
            true,
            chunk_override,
            // #crown-cut-segment: never cut the verdict-shaped α-CROWN
            // backward — cuts are a sweep-only throughput lever.
            None,
            expected_fixed_waves,
        )
    }

    /// Test-only wrapper for the STEP-2b refresh gate (#cgan-alpha-refresh-budget).
    #[cfg(test)]
    pub(crate) fn alpha_refresh_uses_per_node_budget_for_test(
        &self,
        targets: &[String],
        reference_bounds: &HashMap<String, BoundedTensor>,
    ) -> bool {
        self.alpha_refresh_uses_per_node_budget(targets, reference_bounds)
    }

    /// Test-only public wrapper so the cross-module #cgan-alpha-chunk equivalence
    /// test (crown_obj_chunk.rs) can drive the α-CROWN target backward directly
    /// (the real method is `pub(in crate::network::graph_alpha)`). Mirrors the
    /// `crown_ibp_target_can_start_in_patches_for_test` shim pattern.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn propagate_crown_to_node_with_alpha_for_test(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_bounds: &HashMap<String, BoundedTensor>,
        ibp_bounds: &HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_to_node_with_alpha(
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            engine,
            deadline,
        )
    }

    /// Tighten the pre-activation bounds of *dense-fed* ReLUs via a per-target
    /// CROWN backward with the optimized α slopes (#cifar100-fchead).
    ///
    /// Motivation: with `fix_interm_bounds=true` the intermediate bounds returned
    /// by α-CROWN are the (forward-linear / IBP) reference bounds and are never
    /// refined by a CROWN backward. On deep conv ResNets the conv-stack reference
    /// bounds are already tight, but the final *dense* head pre-activation
    /// (cifar100: `Gemm_56`, 2048→100) stays wide — it dominates the residual
    /// ReLU-relaxation slack at the output. Recomputing just those dense head
    /// pre-activations with a backward pass (which optimizes the relaxation for
    /// that specific target, using the tight reference bounds + α for the ReLUs
    /// along the way) tightens them without the O(L²) cost of the full recompute.
    ///
    /// SOUND: each recomputed bound is per-element intersected with the existing
    /// reference bound (`[max(l), min(u)]`, union fallback on the impossible
    /// disjoint case), so the stored bound can only shrink and always encloses the
    /// true reachable pre-activation set. Any per-target failure (unsupported op,
    /// deadline, shape mismatch) leaves that node at its sound reference bound.
    /// Structurally discover dense-head pre-activations: inputs to ReLUs whose
    /// producer is Linear/Gemm. Exporter-assigned node names are deliberately
    /// irrelevant (cifar100's measured `Gemm_56 -> Relu_57` is one instance).
    ///
    /// Kept as the single selector shared by the optimized-alpha FC-head pass and
    /// the bounded heuristic-CROWN root pass so the two lanes cannot silently
    /// drift onto different targets.
    pub(crate) fn fc_head_preactivation_targets(&self, exec_order: &[String]) -> Vec<String> {
        let mut targets: Vec<String> = Vec::new();
        for name in exec_order {
            let Some(node) = self.nodes.get(name) else {
                continue;
            };
            if !matches!(node.layer, Layer::ReLU(_)) {
                continue;
            }
            let Some(pre_name) = node.inputs.first() else {
                continue;
            };
            let is_dense = self
                .nodes
                .get(pre_name)
                .is_some_and(|pre| matches!(pre.layer, Layer::Linear(_)));
            if is_dense && !targets.contains(pre_name) {
                targets.push(pre_name.clone());
            }
        }
        targets
    }

    pub(crate) fn tighten_fc_head_preactivations(
        &self,
        input: &BoundedTensor,
        exec_order: &[String],
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
        bounds: &mut HashMap<String, BoundedTensor>,
    ) {
        let targets = self.fc_head_preactivation_targets(exec_order);
        self.tighten_preactivations_with_alpha(
            input,
            &targets,
            "FC-head tighten",
            alpha_state,
            engine,
            deadline,
            bounds,
        );
    }

    /// Tighten EVERY ReLU pre-activation's bounds with the optimized α slopes
    /// (#root-interm-alpha, dark `NY_ROOT_INTERM_ALPHA=1`). This is the broad
    /// counterpart to [`Self::tighten_fc_head_preactivations`]: where the FC-head
    /// pass refines only the dense-fed ReLU pre-activation (`Gemm_56`), this pass
    /// refines the pre-activations of ALL ReLUs — the conv-stack (BN/Add outputs
    /// feeding conv-block ReLUs) AND the dense head — with the OPTIMIZED root α.
    ///
    /// Motivation (auto_LiRPA parity): with `fix_interm_bounds=true` NY's root
    /// intermediate bounds are the heuristic-α (forward-linear / CROWN-IBP)
    /// reference bounds — the α optimized for the output margin is never applied
    /// to the intermediate pre-activations. auto_LiRPA optimizes the α used to
    /// compute EACH intermediate bound. This measures whether recomputing all
    /// root ReLU pre-activations with the optimized α tightens the downstream
    /// relaxation enough to move the cifar100 worst-subdomain plateau.
    ///
    /// SOUND: identical intersect-only contract as the FC-head pass — each
    /// recomputed bound is per-element intersected into the existing sound
    /// reference bound (shrink-only, union fallback on the impossible disjoint
    /// case), and α only tunes the ReLU lower slope within the sound triangle.
    /// Any per-target failure (unsupported op, deadline, shape/NaN) leaves that
    /// node at its sound reference bound.
    /// #image-node-crown (dark, #root-joint-interm-alpha family): give WIDTH-OUTLIER
    /// nodes a targeted CROWN backward and shrink-intersect the result into the
    /// frozen map. Motivation (measured, cgan): the alpha-refresh targets are
    /// ACTIVATION INPUTS only, so a node that feeds no activation — the generator's
    /// output image ConvTranspose_13 — never gets a CROWN bound; its map entry is
    /// ONE loose interval step off the previous ReLU and blows up 100-160x
    /// (interior generator meanw 0.02-0.14 vs image meanw 15-2500), and every
    /// discriminator bound inherits that width. A targeted backward for the image
    /// node walks the SAME machinery the refresh uses (per-neuron alpha to the
    /// tiny latent box) at a fraction of the cost (its dim is the seed width).
    ///
    /// SOUND: identical contract to `tighten_all_relu_preactivations` — the CROWN
    /// bound is a valid enclosure; intersection with the reference is shrink-only
    /// with per-element union fallback; on deadline/error the reference is kept.
    /// Selection-only heuristics (outlier factor, cap) cannot affect soundness.
    pub(crate) fn tighten_outlier_node_bounds(
        &self,
        input: &BoundedTensor,
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
        bounds: &mut HashMap<String, BoundedTensor>,
    ) -> usize {
        let mean_width = |b: &BoundedTensor| -> f64 {
            let n = b.lower().len().max(1);
            b.lower()
                .iter()
                .zip(b.upper().iter())
                .map(|(&l, &u)| f64::from(u - l))
                .sum::<f64>()
                / n as f64
        };
        let mut widths: Vec<f64> = bounds
            .values()
            .map(mean_width)
            .filter(|w| w.is_finite() && *w > 0.0)
            .collect();
        if widths.len() < 4 {
            return 0;
        }
        widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = widths[widths.len() / 2];
        if !(median.is_finite() && median > 0.0) {
            return 0;
        }
        // Select the AMPLIFIERS, not the widest symptoms: a node whose width is
        // >= 4x its widest parent's is where the enclosure quality is lost (the
        // deep nodes inherit it — CROWN-refreshing THEM is useless while their
        // relaxations are built from the still-loose map). Measured cgan chain:
        // ConvT_13 26.2/1.51 (17x), Conv_16 187/17.9 (10x), Conv_19 401/49.6,
        // Conv_22 2508/261 — exactly the four amplification points. Bounded seed
        // dim, exec-order (upstream first, so later fixes see earlier ones), cap 6.
        let order = match self.exec_order() {
            Ok(o) => o.to_vec(),
            Err(_) => return 0,
        };
        let mut targets: Vec<String> = Vec::new();
        for name in &order {
            if targets.len() >= 6 {
                break;
            }
            let Some(node) = self.nodes.get(name) else {
                continue;
            };
            let Some(b) = bounds.get(name) else {
                continue;
            };
            if b.lower().is_empty() || b.lower().len() > 8192 {
                continue;
            }
            let w = mean_width(b);
            // Absolute floor only (no median gate: on a map that is MOSTLY loose
            // the median itself is huge and masks the true amplifiers — measured
            // cgan: median 49.6 excluded the 17x ConvT_13 and the 10x Conv_16).
            // Env-tunable (#cgan-fwdlin-ref): on a forward-linear-tight base the
            // amplifier chain sits at 0.9-4.6 meanw, so the stale-regime floor
            // of 5.0 breaks the cascade (measured: excluding Conv_16 0.99 /
            // Conv_19 4.58 left Conv_22 at 26 instead of 0.006). The 3x parent
            // ratio below stays the primary guard.
            let floor = std::env::var("NY_IMAGE_NODE_CROWN_FLOOR")
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(5.0);
            if !(w.is_finite() && w >= floor) {
                continue;
            }
            let parent_w = node
                .inputs()
                .iter()
                .filter_map(|p| bounds.get(p).map(mean_width))
                .fold(0.0_f64, f64::max);
            // 3x: the output Gemm's one-interval-step entry measured 3.9x its
            // parent (12391/3190) and slipped a 4x gate; its walk is 1-dim = free.
            if parent_w > 0.0 && w >= 3.0 * parent_w {
                targets.push(name.clone());
            }
        }
        if targets.is_empty() {
            return 0;
        }
        let crown = match self.collect_selected_crown_bounds_with_alpha_mode(
            input,
            &targets,
            bounds,
            alpha_state,
            engine,
            deadline,
            true,
        ) {
            Ok(map) => map,
            Err(_) => return 0, // fail-closed: keep every reference bound
        };
        let mut n_tightened = 0usize;
        let apply = |crown: HashMap<String, BoundedTensor>,
                     bounds: &mut HashMap<String, BoundedTensor>,
                     pass: usize|
         -> usize {
            let mut n = 0usize;
            for (name, crown_bt) in crown {
                let Some(reference) = bounds.get(&name) else {
                    continue;
                };
                if crown_bt.shape() != reference.shape() {
                    continue;
                }
                let (tightened, _disjoint) = reference
                    .intersection_per_element(&crown_bt)
                    .unwrap_or_else(|| (reference.clone(), 0));
                let ref_w = mean_width(reference);
                let new_w = mean_width(&tightened);
                eprintln!(
                    "[image-node-crown] p{pass} '{name}': meanw {ref_w:.4} -> {new_w:.4} (dim={})",
                    tightened.lower().len()
                );
                if (ref_w - new_w) > 1e-6 {
                    n += 1;
                }
                bounds.insert(name, tightened);
            }
            n
        };
        n_tightened += apply(crown, bounds, 1);
        // ITERATED PASSES: each pass's tightening becomes the next pass's
        // relaxation source (the collect relaxes activations off the MAP's
        // pre-activation bounds), so the fix cascades one amplifier deeper per
        // pass (measured cgan chain needs ~4: ConvT_13/Relu_15 -> Conv_16 ->
        // Conv_19 -> Conv_22). Sound: every pass is intersect-only. Stops on a
        // no-improvement pass, deadline, or the pass cap.
        let mut pass = 2usize;
        loop {
            if pass > 6 || n_tightened == 0 {
                break;
            }
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    break;
                }
            }
            let Ok(crown_n) = self.collect_selected_crown_bounds_with_alpha_mode(
                input,
                &targets,
                bounds,
                alpha_state,
                engine,
                deadline,
                true,
            ) else {
                break;
            };
            let n = apply(crown_n, bounds, pass);
            if n == 0 {
                break;
            }
            n_tightened += n;
            pass += 1;
        }
        n_tightened
    }

    pub(crate) fn tighten_all_relu_preactivations(
        &self,
        input: &BoundedTensor,
        exec_order: &[String],
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
        bounds: &mut HashMap<String, BoundedTensor>,
    ) {
        // Every ReLU's single input node is its pre-activation target. Preserve
        // exec order (targets computed against a frozen reference snapshot, so
        // order does not affect the bounds) and de-duplicate.
        let mut targets: Vec<String> = Vec::new();
        for name in exec_order {
            let Some(node) = self.nodes.get(name) else {
                continue;
            };
            if !matches!(node.layer, Layer::ReLU(_)) {
                continue;
            }
            let Some(pre_name) = node.inputs.first() else {
                continue;
            };
            if !targets.contains(pre_name) {
                targets.push(pre_name.clone());
            }
        }
        self.tighten_preactivations_with_alpha(
            input,
            &targets,
            "root-interm-α tighten",
            alpha_state,
            engine,
            deadline,
            bounds,
        );
    }

    /// Shared core for the α-optimized pre-activation tightening passes
    /// ([`Self::tighten_fc_head_preactivations`] and
    /// [`Self::tighten_all_relu_preactivations`]): recompute each `targets`
    /// pre-activation bound with a per-target α-CROWN backward (using the
    /// optimized slopes + a frozen reference snapshot) and per-element intersect
    /// it into `bounds` (shrink-only, union fallback). `label` names the pass in
    /// debug logs. See the callers for the soundness contract.
    #[allow(clippy::too_many_arguments)]
    fn tighten_preactivations_with_alpha(
        &self,
        input: &BoundedTensor,
        targets: &[String],
        label: &str,
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
        bounds: &mut HashMap<String, BoundedTensor>,
    ) {
        if targets.is_empty() {
            return;
        }
        debug!(
            "{label}: {} pre-activation target(s) {:?}",
            targets.len(),
            targets
        );

        // Reference snapshot: relax the ReLUs along the backward path with the
        // pre-tightening reference bounds so each target is computed against the
        // same intermediate bounds (order-independent, no mid-loop feedback).
        let reference = bounds.clone();
        let empty: HashMap<String, BoundedTensor> = HashMap::new();
        // #root-interm-alpha DIAGNOSTIC: accumulate the total mean-width the pass
        // removes so the measurement can confirm the gate fired AND quantify how
        // much it tightened (a zero total = the α-optimized backward matched the
        // reference bound to ULP, i.e. no tightening available). eprintln only
        // reached under the dark FCHEAD/RIA opt-in gates (this core is never
        // called on the default path), so it does not affect default runs.
        let mean_width = |bt: &BoundedTensor| -> f64 {
            let (mut s, mut n) = (0.0f64, 0usize);
            for (&l, &u) in bt.lower().iter().zip(bt.upper().iter()) {
                if (u - l).is_finite() {
                    s += (u - l) as f64;
                    n += 1;
                }
            }
            if n > 0 {
                s / n as f64
            } else {
                0.0
            }
        };
        let mut n_ok = 0usize;
        let mut n_moved = 0usize;
        let mut total_ref_w = 0.0f64;
        let mut total_new_w = 0.0f64;
        for target in targets {
            let Some(reference_bound) = reference.get(target) else {
                continue;
            };
            let crown = match self.propagate_crown_to_node_with_alpha(
                input,
                target,
                &empty,
                &reference,
                alpha_state,
                engine,
                deadline,
            ) {
                Ok(b) => b,
                Err(
                    e @ (NyError::UnsupportedOp(_)
                    | NyError::UnsupportedConfiguration(_)
                    | NyError::ShapeMismatch { .. }
                    | NyError::CpuMemoryExceeded { .. }
                    | NyError::DeadlineExceeded(_)),
                ) => {
                    // Deadline is the common case here: the warmup budget is spent
                    // and only a small grace slice was granted. Keep the sound
                    // reference bound for this target and move on.
                    debug!(
                        "{label}: node '{}' CROWN backward unavailable ({}), keeping reference bound",
                        target, e
                    );
                    continue;
                }
                Err(NyError::LayerError { source, .. })
                    if matches!(
                        source.as_ref(),
                        NyError::SoundnessRefusal(_)
                            | NyError::NumericalInstability(_)
                            | NyError::InternalError(_)
                    ) =>
                {
                    debug!(
                        "{label}: node '{}' backward soundness/internal error, keeping reference bound",
                        target
                    );
                    continue;
                }
                Err(_) => continue,
            };
            if crown.shape() != reference_bound.shape() {
                continue;
            }
            let (tightened, disjoint) = reference_bound
                .intersection_per_element(&crown)
                .unwrap_or_else(|| (reference_bound.clone(), 0));
            if disjoint > 0 {
                debug!(
                    "{label}: '{}' {} disjoint elements, used union fallback (still sound)",
                    target, disjoint
                );
            }
            let ref_w = mean_width(reference_bound);
            let new_w = mean_width(&tightened);
            n_ok += 1;
            total_ref_w += ref_w;
            total_new_w += new_w;
            if (ref_w - new_w) > 1e-6 {
                n_moved += 1;
            }
            if tracing::enabled!(tracing::Level::DEBUG) {
                debug!(
                    "{label}: '{}' refined via α-CROWN backward: ref_meanw={:.4} -> meanw={:.4}",
                    target, ref_w, new_w
                );
            }
            bounds.insert(target.clone(), tightened);
        }
        eprintln!(
            "[{label}] targets={} ok={} moved={} total_ref_meanw={:.4} -> total_meanw={:.4} (reduction={:.4})",
            targets.len(),
            n_ok,
            n_moved,
            total_ref_w,
            total_new_w,
            total_ref_w - total_new_w
        );
    }

    /// STABILIZE-AND-FIX tighten lane (#stabilize, dark `NY_STABILIZE=<secs>`):
    /// recompute each ranked pre-activation target with a per-target α-CROWN
    /// backward against a FROZEN reference snapshot of `bounds`, then
    /// per-element intersect the result into the stored entry (shrink-only,
    /// union fallback on the impossible-disjoint case). Differs from
    /// [`Self::tighten_preactivations_with_alpha`] only in scheduling: each
    /// target gets an equal share of the remaining budget
    /// (`remaining / remaining_targets`, the `budget_policy` per-node-window
    /// pattern) instead of one flat deadline, so one slow target cannot starve
    /// the rest of the ranked list.
    ///
    /// SOUND: every write is `reference ∩ certified-backward` of two sound
    /// enclosures of the same reachable set over the root box. The error
    /// taxonomy is copied from `tighten_preactivations_with_alpha`:
    /// UnsupportedOp / UnsupportedConfiguration / ShapeMismatch /
    /// CpuMemoryExceeded / DeadlineExceeded and LayerError-wrapped
    /// SoundnessRefusal / NumericalInstability / InternalError all KEEP the
    /// sound reference bound for that target and move on — no error is ever
    /// silently folded into a stored bound. Budget expiry yields partial work
    /// with every completed merge sound.
    ///
    /// Returns the number of targets whose stored entry was refreshed.
    pub(crate) fn stabilize_tighten_targets_with_alpha(
        &self,
        input: &BoundedTensor,
        targets: &[String],
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
        budget_deadline: Instant,
        bounds: &mut HashMap<String, BoundedTensor>,
    ) -> usize {
        if targets.is_empty() {
            return 0;
        }
        // Frozen reference snapshot: every target is computed against the same
        // pre-round intermediate bounds (order-independent, no mid-loop
        // feedback), exactly like `tighten_preactivations_with_alpha`.
        let reference = bounds.clone();
        let empty: HashMap<String, BoundedTensor> = HashMap::new();
        let mut merged = 0usize;
        for (pos, target) in targets.iter().enumerate() {
            let now = Instant::now();
            if now >= budget_deadline {
                break; // partial work: every completed merge above is sound
            }
            // Equal-share per-node window over the remaining budget; the last
            // target inherits whatever remains.
            let remaining_targets = targets.len().saturating_sub(pos).max(1) as u32;
            let node_deadline = Some(now + (budget_deadline - now) / remaining_targets);
            let Some(reference_bound) = reference.get(target) else {
                continue;
            };
            let crown = match self.propagate_crown_to_node_with_alpha(
                input,
                target,
                &empty,
                &reference,
                alpha_state,
                engine,
                node_deadline,
            ) {
                Ok(b) => b,
                Err(
                    e @ (NyError::UnsupportedOp(_)
                    | NyError::UnsupportedConfiguration(_)
                    | NyError::ShapeMismatch { .. }
                    | NyError::CpuMemoryExceeded { .. }
                    | NyError::DeadlineExceeded(_)),
                ) => {
                    debug!(
                        "stabilize tighten: node '{}' CROWN backward unavailable ({}), \
                         keeping reference bound",
                        target, e
                    );
                    continue;
                }
                Err(NyError::LayerError { source, .. })
                    if matches!(
                        source.as_ref(),
                        NyError::SoundnessRefusal(_)
                            | NyError::NumericalInstability(_)
                            | NyError::InternalError(_)
                    ) =>
                {
                    debug!(
                        "stabilize tighten: node '{}' backward soundness/internal error, \
                         keeping reference bound",
                        target
                    );
                    continue;
                }
                Err(_) => continue,
            };
            if crown.shape() != reference_bound.shape() {
                continue;
            }
            // Shrink-only merge (NaN ⇒ None ⇒ keep reference; disjoint ⇒ union).
            let Some((tightened, disjoint)) = reference_bound.intersection_per_element(&crown)
            else {
                continue;
            };
            if disjoint > 0 {
                debug!(
                    "stabilize tighten: '{}' {} disjoint elements, used union fallback (still sound)",
                    target, disjoint
                );
            }
            bounds.insert(target.clone(), tightened);
            merged += 1;
        }
        merged
    }
}
