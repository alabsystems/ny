// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Constraint-aware CROWN propagation for graph branch-and-bound.
//!
//! Implements the forward–backward bound computation that incorporates ReLU split
//! constraints from the BaB search tree. The forward pass tightens intermediate
//! bounds using active/inactive neuron constraints; the backward pass computes
//! CROWN linear relaxation coefficients with optional β (Lagrangian) contributions.
//!
//! Submodules:
//! - `backward`: backward traversal with constraint-aware ReLU relaxation
//! - `lookups`: pre-constraint application and constraint index construction

pub(in crate::beta_crown::engine::graph::propagation) mod backward;
mod clip_runtime;
mod lookups;
mod patches;
mod spec_matrix;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use ny_core::{nan_propagating_max, nan_propagating_min, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::domain::{GraphCrownContext, NodeBoundsView};
use crate::beta_crown::state::GraphBetaState;
use crate::layers::common::BoundPropagation;
use crate::{GraphNetwork, Layer, NETWORK_INPUT};

use super::super::super::BetaCrownVerifier;
use crate::beta_crown::engine::graph::{DomainCrownResult, DomainCrownResultWithIntermediates};

use backward::{BackwardMode, BackwardParams};
use lookups::{apply_genbab_pre_constraints, apply_pre_constraints, build_constraint_lookups};
pub(in crate::beta_crown::engine::graph::propagation) use patches::ConstrainedPatchesPolicy;

fn ensure_constrained_propagation_deadline(
    deadline: Option<std::time::Instant>,
    stage: &str,
) -> Result<()> {
    if deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
        return Err(NyError::DeadlineExceeded(format!(
            "constrained CROWN: deadline exceeded {stage}"
        )));
    }
    Ok(())
}

/// #cone-delta gate. `NY_CONE_REFRESH=1` enables delta seeding of the
/// constrained forward pass (recompute only the cone of the constraints added
/// since the inherited map was last fixpointed, instead of the full split
/// history's cone). Unset/anything-else keeps the full-history seeds —
/// bit-for-bit today's behavior. Dark by default; the flip decision comes from
/// the increment-4 A/B measurement, not from this code.
fn cone_refresh_enabled() -> bool {
    matches!(std::env::var("NY_CONE_REFRESH").ok().as_deref(), Some("1"))
}

impl BetaCrownVerifier {
    pub(crate) fn propagate_crown_with_graph_constraints(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: Option<&GraphBetaState>,
        objective: Option<&[f32]>,
    ) -> Result<DomainCrownResult> {
        let (output_bounds, bounds_cache, _captured_la) = self
            .propagate_crown_with_graph_constraints_with_cache(
                graph, input, context, beta_state, objective, None, false,
            )?;
        Ok((output_bounds, bounds_cache))
    }

    // Justification: constrained graph propagation must thread graph/input/context,
    // optional beta state/objective, and optional cache capture in one call.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_with_graph_constraints_with_cache(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: Option<&GraphBetaState>,
        objective: Option<&[f32]>,
        seed_cache: Option<&CachedLinearBounds>,
        capture_linear_bounds: bool,
    ) -> Result<(
        BoundedTensor,
        std::collections::HashMap<String, Arc<BoundedTensor>>,
        Option<CachedLinearBounds>,
    )> {
        let deadline = self.effective_graph_bab_deadline();
        ensure_constrained_propagation_deadline(
            deadline,
            "before constrained forward preparation",
        )?;
        let (mut bounds_cache, constrained_input, exec_order) =
            self.prepare_constrained_graph_bounds(graph, input, context, beta_state, objective)?;

        ensure_constrained_propagation_deadline(deadline, "before Complete Clip")?;
        self.maybe_apply_complete_clip_root_bank(
            graph,
            context,
            beta_state,
            objective,
            None,
            &constrained_input,
            &exec_order,
            &mut bounds_cache,
        );
        ensure_constrained_propagation_deadline(
            deadline,
            "after Complete Clip and before constrained backward preparation",
        )?;

        // #1817 diagnostic: dump forward bounds after constraint tightening when fully constrained
        if tracing::enabled!(tracing::Level::DEBUG) {
            let total_relu_neurons: usize = exec_order
                .iter()
                .filter_map(|n| {
                    let node = graph.nodes.get(n)?;
                    if matches!(node.layer, Layer::ReLU(_)) {
                        Some(bounds_cache.get(n).map(|b| b.len()).unwrap_or(0))
                    } else {
                        None
                    }
                })
                .sum();
            let n_constraints = context.history.constraints.len();
            if n_constraints == total_relu_neurons && n_constraints > 0 {
                for n in &exec_order {
                    if let Some(b) = bounds_cache.get(n) {
                        let lower: Vec<f32> = b.flatten().lower().iter().copied().collect();
                        let upper: Vec<f32> = b.flatten().upper().iter().copied().collect();
                        debug!("[#1817 fwd] {}: lower={:?}, upper={:?}", n, lower, upper);
                    }
                }
            }
        }
        // #1817: log exec order for debugging
        if context.history.constraints.len() >= 12 {
            debug!("[#1817 exec] order: {:?}", exec_order);
        }

        ensure_constrained_propagation_deadline(deadline, "before constrained backward dispatch")?;
        // Delegate to shared backward CROWN core (standard mode: no intermediate storage).
        let params = BackwardParams {
            graph,
            constrained_input: &constrained_input,
            exec_order: &exec_order,
            context,
            beta_state,
            objective,
            spec_matrix: None,
            seed_cache,
            capture_linear_bounds,
            deadline, // #3795: thread BaB deadline
            patches_policy: ConstrainedPatchesPolicy::for_engine(context.engine),
        };
        let result =
            self.backward_crown_constrained(&params, &mut bounds_cache, BackwardMode::Standard)?;

        Ok((result.output_bounds, bounds_cache, result.captured_la))
    }

    /// Shared constraint-aware forward bound tightening for all constraint propagation pipelines.
    ///
    /// Both `propagate_crown_with_graph_constraints` and
    /// `propagate_crown_with_graph_constraints_storing_intermediates` delegate their forward
    /// pass to this function. Also called directly by the batched backward path.
    ///
    /// `delta_seeds` (#cone-delta): the pre-activation nodes of the constraints
    /// added to `history` since `base_bounds` was last fixpointed by this same
    /// routine (`GraphBabDomain::delta_pre_nodes`). `None` = no delta known.
    /// Only consulted behind `NY_CONE_REFRESH=1` and a battery of fail-closed
    /// checks — see the seed-selection block below; otherwise ignored.
    pub(in crate::beta_crown::engine::graph) fn compute_constrained_forward_bounds(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        history: &GraphSplitHistory,
        base_bounds: Option<&std::collections::HashMap<String, Arc<BoundedTensor>>>,
        delta_seeds: Option<&[String]>,
    ) -> Result<(
        std::collections::HashMap<String, Arc<BoundedTensor>>,
        BoundedTensor,
    )> {
        self.compute_constrained_forward_bounds_inner(
            graph,
            input,
            history,
            base_bounds,
            delta_seeds,
            true,
        )
    }

    /// Read-only compatibility face for a provenance-tracked node-bound map.
    pub(in crate::beta_crown::engine::graph) fn compute_constrained_forward_bounds_from_view(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        history: &GraphSplitHistory,
        base_bounds: Option<NodeBoundsView<'_>>,
        delta_seeds: Option<&[String]>,
    ) -> Result<(
        std::collections::HashMap<String, Arc<BoundedTensor>>,
        BoundedTensor,
    )> {
        self.compute_constrained_forward_bounds_view_inner(
            graph,
            input,
            history,
            base_bounds,
            delta_seeds,
            true,
        )
    }

    /// Inner implementation of [`Self::compute_constrained_forward_bounds`].
    ///
    /// `enable_upstream_cache` gates the upstream-bound inheritance optimization.
    /// Production callers always set it to `true`; the equivalence tests set it
    /// to `false` to obtain the full-recomputation reference (every node
    /// re-propagated from the inherited seed) and assert it equals the cached
    /// result element-wise.
    pub(in crate::beta_crown::engine::graph) fn compute_constrained_forward_bounds_inner(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        history: &GraphSplitHistory,
        base_bounds: Option<&std::collections::HashMap<String, Arc<BoundedTensor>>>,
        delta_seeds: Option<&[String]>,
        enable_upstream_cache: bool,
    ) -> Result<(
        std::collections::HashMap<String, Arc<BoundedTensor>>,
        BoundedTensor,
    )> {
        self.compute_constrained_forward_bounds_view_inner(
            graph,
            input,
            history,
            base_bounds.map(NodeBoundsView::from_hash_map),
            delta_seeds,
            enable_upstream_cache,
        )
    }

    fn compute_constrained_forward_bounds_view_inner(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        history: &GraphSplitHistory,
        base_bounds: Option<NodeBoundsView<'_>>,
        delta_seeds: Option<&[String]>,
        enable_upstream_cache: bool,
    ) -> Result<(
        std::collections::HashMap<String, Arc<BoundedTensor>>,
        BoundedTensor,
    )> {
        let deadline = self.effective_graph_bab_deadline();
        // #layer-deadline-suppression: the per-node LOOP keeps `deadline`; the
        // layer kernels get `None` only inside an explicitly scoped advisory
        // caller. See `suppress_layer_deadline_scoped` for the soundness
        // argument (a-priori gamma certificate, looser-or-equal, never tighter).
        let layer_deadline = if self
            .complete_clip_deadline_overrides
            .layer_deadline_suppressed()
        {
            None
        } else {
            deadline
        };
        ensure_constrained_propagation_deadline(deadline, "before constrained forward bounds")?;
        let exec_order = graph.exec_order()?;
        let reusing_inherited_bounds = base_bounds.is_some();

        // Use base bounds if provided, otherwise compute IBP bounds.
        //
        // #cone-delta increment 2 (Arc-preserving cache): seeding from a parent
        // map is `Arc::clone` per entry — NOT a deep tensor copy. Entries the
        // loop below never replaces (out-of-cone nodes under the upstream-cache
        // skip) therefore remain pointer-shared with the parent's map; every
        // entry the loop does replace is inserted as a fresh `Arc`. Sharing is
        // safe because `BoundedTensor` has no interior mutability and no path
        // mutates a cache entry in place (entries are only ever replaced).
        let mut bounds_cache: std::collections::HashMap<String, Arc<BoundedTensor>> =
            if let Some(base) = base_bounds {
                base.iter()
                    .map(|(k, v)| (k.clone(), Arc::clone(v)))
                    .collect()
            } else {
                let initial_bounds = if deadline.is_some() {
                    graph.collect_node_bounds_with_engine_and_deadline(input, None, deadline)?
                } else {
                    graph.collect_node_bounds(input)?
                };
                initial_bounds
                    .into_iter()
                    .map(|(k, v)| (k, Arc::new(v)))
                    .collect()
            };

        // #2399: Build lookups from both ReLU and GenBaB constraints so the forward pass
        // tightens bounds at general nonlinearities (GeLU, Sigmoid, Tanh), not just ReLU.
        let lookups =
            build_constraint_lookups(&history.constraints, &history.genbab_constraints, graph)?;

        // Apply constraints to input
        let mut constrained_input = input.clone();
        if let Some(cons) = lookups.pre.get(NETWORK_INPUT) {
            constrained_input = apply_pre_constraints(&constrained_input, cons)?;
        }
        if let Some(cons) = lookups.pre_genbab.get(NETWORK_INPUT) {
            constrained_input = apply_genbab_pre_constraints(&constrained_input, cons)?;
        }

        // Upstream-bound inheritance (graph BaB per-domain caching).
        //
        // When we are reusing a parent domain's inherited node bounds
        // (`base_bounds` provided) and there is at least one split constraint,
        // a node's re-propagated bound can only DIFFER from its inherited seed
        // if the node lies downstream of a node whose bounds the split directly
        // tightens. For any node that is *not* a (transitive) descendant of such
        // a node, both the node and all of its inputs keep their parent values,
        // so the IBP re-propagation + per-element intersection performed below is
        // provably idempotent and returns exactly the seeded (parent) bound. We
        // therefore skip the work for those nodes and keep their seed verbatim.
        //
        // SOUNDNESS — choice of seed set. A ReLU/GenBaB split constraint is
        // applied in this forward pass as a *pre-activation* tightening: it
        // modifies the bounds of the constrained node's PRE-ACTIVATION node (the
        // input feeding the ReLU/Sign/nonlinearity — see `apply_pre_constraints`
        // / `apply_genbab_pre_constraints` and `build_constraint_lookups`, where
        // the `pre`/`pre_genbab` maps are keyed by pre-activation node name).
        // The earliest node whose bounds change is therefore that PRE-ACTIVATION
        // node, NOT the ReLU node itself. Seeding downstream-reachability from
        // the pre-activation nodes (the exact keys of `lookups.pre` and
        // `lookups.pre_genbab`) captures every node that can change, including
        // the pre-activation node, the ReLU/nonlinear node, and everything after.
        //
        // `descendants_inclusive` treats a `NETWORK_INPUT` seed (or any
        // unresolved node) as reaching everything, so if the split tightens the
        // network input we recompute the whole graph. Anything we cannot prove
        // unaffected is recomputed.
        //
        // The skip is gated on `reusing_inherited_bounds` (we must have a parent
        // seed to reuse), `enable_upstream_cache`, and a non-empty seed set (when
        // there are no constraints the original full pass is preserved, matching
        // prior behavior for non-BaB callers such as the root final propagation).
        let affected_downstream: Option<std::collections::HashSet<String>> = {
            // Seed set = every node whose bounds a constraint directly modifies:
            // the pre-activation node of each ReLU and GenBaB constraint.
            let mut seeds: Vec<String> = lookups
                .pre
                .keys()
                .chain(lookups.pre_genbab.keys())
                .cloned()
                .collect();
            if enable_upstream_cache && reusing_inherited_bounds {
                // #cone-delta: when the caller supplied the delta since the
                // inherited map's last fixpoint AND every fail-closed condition
                // holds, seed the recompute cone from the delta alone — the
                // full-history clamps below still apply to every node the loop
                // visits, only the SEED SET shrinks. An empty delta (re-bounding
                // an already-fixpointed domain) yields an empty cone: the whole
                // inherited map is reused verbatim. Any condition failing falls
                // back to the full-history seeds — the existing tested behavior.
                if let Some(delta) =
                    self.select_delta_seeds(graph, delta_seeds, &seeds, exec_order, base_bounds)
                {
                    let mut delta = delta.to_vec();
                    delta.sort();
                    delta.dedup();
                    Some(graph.descendants_inclusive(&delta)?)
                } else if !seeds.is_empty() {
                    // `descendants_inclusive` handles a NETWORK_INPUT seed by
                    // marking the whole graph affected, so input-level splits
                    // recompute fully.
                    seeds.sort();
                    seeds.dedup();
                    Some(graph.descendants_inclusive(&seeds)?)
                } else {
                    None
                }
            } else {
                None
            }
        };

        // Apply constraint tightening to bounds cache
        for node_name in exec_order {
            ensure_constrained_propagation_deadline(
                deadline,
                &format!("before constrained forward node '{node_name}'"),
            )?;
            // Reuse the inherited (parent) bound verbatim for nodes the split
            // provably cannot affect. The seed already lives in `bounds_cache`.
            if let Some(downstream) = &affected_downstream {
                if !downstream.contains(node_name.as_str())
                    && bounds_cache.contains_key(node_name.as_str())
                {
                    continue;
                }
            }
            let node = graph
                .nodes
                .get(node_name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            let current_bounds = bounds_cache.get(node_name.as_str()).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "CROWN-IBP bounds not found for node '{}'",
                    node_name
                ))
            })?;

            let mut output_bounds = if matches!(node.layer, Layer::ReLU(_)) {
                let first_input = node.inputs.first().ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Constrained CROWN-IBP failed at '{}' ({}): node has no inputs",
                        node_name,
                        node.layer.layer_type()
                    ))
                })?;
                let pre_activation: &BoundedTensor = if first_input == NETWORK_INPUT {
                    &constrained_input
                } else {
                    bounds_cache
                        .get(first_input)
                        .map(|a| a.as_ref())
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Pre-activation bounds for {} not found",
                                first_input
                            ))
                        })?
                };

                let pre_flat = pre_activation.flatten();
                let shape = pre_activation.shape().to_vec();
                let mut lower = pre_flat.lower().clone();
                let mut upper = pre_flat.upper().clone();

                // Flatten the inherited (post-activation) CROWN-IBP bounds ONCE.
                // It is loop-invariant across neurons; flattening per-neuron made
                // the unconstrained-ReLU branch below O(N²) in allocations for a
                // ReLU with N neurons (the conv/resnet per-domain hot path). The
                // flattened tensor and its per-index reads are identical regardless
                // of which iteration produces it.
                let crown_bounds = current_bounds.flatten();

                // ReLU preserves shape: pre-activation and current (post-activation)
                // bounds must have the same flat length. (#2671, #2920 WP-B)
                if pre_flat.len() != crown_bounds.len() {
                    return Err(NyError::InternalError(format!(
                        "pre_flat vs current_bounds shape mismatch at ReLU node '{}': \
                         pre_flat.len()={} != current_bounds.flatten().len()={}",
                        node_name,
                        pre_flat.len(),
                        crown_bounds.len()
                    )));
                }
                let relu_cons = lookups.by_relu.get(node_name);
                for neuron_idx in 0..pre_flat.len() {
                    let l = pre_flat.lower()[[neuron_idx]];
                    let u = pre_flat.upper()[[neuron_idx]];

                    let constrained = relu_cons.and_then(|m| m.get(&neuron_idx).copied());
                    match constrained {
                        Some(true) => {
                            if u < 0.0 {
                                // #2926: Single-constraint infeasibility — active constraint
                                // on provably-negative neuron. Domain is empty.
                                return Err(NyError::InfeasibleDomain(format!(
                                    "active ReLU constraint at node '{}' idx={} but pre_u={} < 0",
                                    node_name, neuron_idx, u
                                )));
                            }
                            // NaN-safe: propagate NaN instead of silently clamping to 0.0 (#2643)
                            lower[[neuron_idx]] = nan_propagating_max(l, 0.0);
                            upper[[neuron_idx]] = u;
                        }
                        Some(false) => {
                            if l > 0.0 {
                                // #2926: Single-constraint infeasibility — inactive constraint
                                // on provably-positive neuron. Domain is empty.
                                return Err(NyError::InfeasibleDomain(format!(
                                    "inactive ReLU constraint at node '{}' idx={} but pre_l={} > 0",
                                    node_name, neuron_idx, l
                                )));
                            }
                            lower[[neuron_idx]] = 0.0;
                            upper[[neuron_idx]] = 0.0;
                        }
                        None => {
                            // Unconstrained: use the fresh child-domain ReLU bound, and only
                            // intersect with inherited cache entries when they agree. (#1817)
                            let relu_lower = nan_propagating_max(l, 0.0);
                            let relu_upper = nan_propagating_max(u, 0.0);
                            let crown_l = crown_bounds.lower()[[neuron_idx]];
                            let crown_u = crown_bounds.upper()[[neuron_idx]];
                            let tightened_lower = nan_propagating_max(relu_lower, crown_l);
                            let tightened_upper = nan_propagating_min(relu_upper, crown_u);
                            if reusing_inherited_bounds && tightened_lower > tightened_upper {
                                debug!(
                                    node = %node_name,
                                    neuron_idx,
                                    "Constraint forward: inherited ReLU cache conflicted with fresh child bounds; keeping re-propagated interval"
                                );
                                lower[[neuron_idx]] = relu_lower;
                                upper[[neuron_idx]] = relu_upper;
                            } else {
                                lower[[neuron_idx]] = tightened_lower;
                                upper[[neuron_idx]] = tightened_upper;
                            }
                        }
                    }
                }

                // #2926: Check for multi-constraint infeasibility before constructing
                // BoundedTensor. When constraint tightening + intersection produces
                // lower > upper for any element, the domain is empty. Return
                // InfeasibleDomain so the BaB loop treats this as verified, not failed.
                let has_inverted = lower.iter().zip(upper.iter()).any(|(&l, &u)| l > u);
                if has_inverted {
                    return Err(NyError::InfeasibleDomain(format!(
                        "constraint interaction at ReLU node '{}': tightened bounds have lower > upper",
                        node_name,
                    )));
                }

                let lower_arr = lower
                    .into_shape_clone(ndarray::IxDyn(&shape))
                    .map_err(|e| NyError::InvalidSpec(format!("shape error: {}", e)))?;
                let upper_arr = upper
                    .into_shape_clone(ndarray::IxDyn(&shape))
                    .map_err(|e| NyError::InvalidSpec(format!("shape error: {}", e)))?;
                BoundedTensor::new(lower_arr, upper_arr)?
            } else {
                // Re-propagate forward through non-ReLU layers using constraint-tightened
                // input bounds. Without this, linear layers after constrained ReLUs keep their
                // original (wide) IBP bounds, causing CROWN backward to see unstable neurons
                // that should be fully determined. (#1817)
                let repropagated = if node.inputs.is_empty() {
                    current_bounds.as_ref().clone()
                } else if matches!(node.layer, Layer::Concat(_)) {
                    // Concat is n-ary — must be checked before is_binary() to avoid
                    // routing through propagate_ibp_binary which only uses 2 inputs.
                    // Matches the dispatch order in graph_ibp.rs and detailed.rs. (#2398)
                    if let Layer::Concat(concat) = &node.layer {
                        // Reconstruct full input list: constant_inputs indexed by
                        // original ONNX order, graph edges for non-constant slots.
                        // Cache entries are `Arc::clone`d (cheap); constants and the
                        // constrained input are wrapped in fresh Arcs to unify the type.
                        let owned_inputs: Vec<Arc<BoundedTensor>> = if let Some(ref ci) =
                            concat.constant_inputs
                        {
                            let mut graph_idx = 0;
                            ci.iter()
                                .map(|const_opt| {
                                    if let Some(constant) = const_opt {
                                        Ok(Arc::new(constant.clone()))
                                    } else {
                                        let inp_name =
                                            node.inputs.get(graph_idx).ok_or_else(|| {
                                                NyError::InternalError(format!(
                                            "Concat '{}': ran out of graph inputs at graph_idx {}",
                                            node_name, graph_idx
                                        ))
                                            })?;
                                        graph_idx += 1;
                                        if inp_name == NETWORK_INPUT {
                                            Ok(Arc::new(constrained_input.clone()))
                                        } else {
                                            bounds_cache.get(inp_name).cloned().ok_or_else(|| {
                                                NyError::InvalidSpec(format!(
                                                    "Bounds for {} not found",
                                                    inp_name
                                                ))
                                            })
                                        }
                                    }
                                })
                                .collect::<std::result::Result<Vec<_>, _>>()?
                        } else {
                            node.inputs
                                .iter()
                                .map(|inp_name| {
                                    if inp_name == NETWORK_INPUT {
                                        Ok(Arc::new(constrained_input.clone()))
                                    } else {
                                        bounds_cache.get(inp_name).cloned().ok_or_else(|| {
                                            NyError::InvalidSpec(format!(
                                                "Bounds for {} not found",
                                                inp_name
                                            ))
                                        })
                                    }
                                })
                                .collect::<std::result::Result<Vec<_>, _>>()?
                        };
                        let refs: Vec<&BoundedTensor> =
                            owned_inputs.iter().map(|a| a.as_ref()).collect();
                        concat.propagate_ibp_nary(&refs).map_err(|e| {
                            NyError::InternalError(format!(
                                "Constraint forward: concat IBP failed at node '{}': {}",
                                node_name, e
                            ))
                        })?
                    } else {
                        current_bounds.as_ref().clone()
                    }
                } else if node.layer.is_binary() {
                    let (input_a_name, input_b_name) = node.require_binary_inputs().map_err(|_| {
                        NyError::InvalidSpec(format!(
                            "Constraint forward: binary node '{}' ({}) requires 2 inputs but has {}",
                            node_name,
                            node.layer.layer_type(),
                            node.inputs.len()
                        ))
                    })?;
                    // #cone-delta: fail closed on a missing input entry. The old
                    // `unwrap_or(current_bounds)` silently substituted THIS
                    // node's own bounds for a missing input — silent-wrong on
                    // any partial map. Unreachable on well-formed graphs (the
                    // input node's own lookup above errors first), so erroring
                    // here is behavior-identical where it matters and a clean
                    // failure where it does not.
                    let input_a: &BoundedTensor = if input_a_name == NETWORK_INPUT {
                        &constrained_input
                    } else {
                        bounds_cache.get(input_a_name).map(|a| a.as_ref()).ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Constraint forward: bounds for input '{}' of binary node '{}' not found",
                                input_a_name, node_name
                            ))
                        })?
                    };
                    let input_b: &BoundedTensor = if input_b_name == NETWORK_INPUT {
                        &constrained_input
                    } else {
                        bounds_cache.get(input_b_name).map(|a| a.as_ref()).ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Constraint forward: bounds for input '{}' of binary node '{}' not found",
                                input_b_name, node_name
                            ))
                        })?
                    };
                    node.layer
                        .propagate_ibp_binary(input_a, input_b)
                        .map_err(|e| {
                            NyError::InternalError(format!(
                                "Constraint forward: binary IBP failed at node '{}' ({}): {}",
                                node_name,
                                node.layer.layer_type(),
                                e
                            ))
                        })?
                } else {
                    // Unary layer (Linear, BatchNorm, Flatten, activation, etc.).
                    // Surface the real arity on a malformed multi-input unary node
                    // (e.g. a Sigmoid/SkipMerge wired with 2 inputs) instead of
                    // masking every failure as "has no inputs" (#1840).
                    let input_name = node.require_unary_input().map_err(|_| {
                        NyError::InvalidSpec(format!(
                            "Constraint forward: {} node {} expects exactly 1 input, got {}",
                            node.layer.layer_type(),
                            node_name,
                            node.inputs.len()
                        ))
                    })?;
                    // #cone-delta: fail closed on a missing input entry (see the
                    // binary-input note above).
                    let node_input: &BoundedTensor = if input_name == NETWORK_INPUT {
                        &constrained_input
                    } else {
                        bounds_cache
                            .get(input_name)
                            .map(|a| a.as_ref())
                            .ok_or_else(|| {
                                NyError::InvalidSpec(format!(
                                "Constraint forward: bounds for input '{}' of node '{}' not found",
                                input_name, node_name
                            ))
                            })?
                    };
                    let propagated = match &node.layer {
                        // Constrained child bounds feed later ReLU phase decisions and
                        // are intersected into the inherited sound enclosure. Plain
                        // f32 Conv2d IBP can under-enclose under cancellation, so this
                        // must retain the certified Higham/abssum widening as well as
                        // the finite-deadline pollable route.
                        // #layer-deadline-suppression: a finite LAYER deadline
                        // routes Conv2d to the certified-f64 five-deep scalar loop
                        // (IxDyn indexing, poll every 4096 taps) instead of im2col
                        // + faer GEMM -- measured at ~91x on this model family.
                        // `None` keeps a certificate: the sound IBP then uses the
                        // f32 GEMM PLUS the |W|*max(|l|,|u|) abssum pass and the
                        // gamma_{K+2}^{f32}*S_safe + 2u*|y| outward widening, i.e.
                        // the a-priori certificate rather than the measured-f64
                        // one. Sound and LOOSER-OR-EQUAL, never tighter.
                        Layer::Conv2d(conv) => conv.propagate_ibp_sound_with_engine_and_deadline(
                            node_input,
                            None,
                            layer_deadline,
                        ),
                        // N-D Linear IBP otherwise enters four opaque faer
                        // products. The finite helper caps all geometry and
                        // polls its direct f64 contractions.
                        Layer::Linear(linear) => linear.propagate_ibp_with_engine_and_deadline(
                            node_input,
                            None,
                            layer_deadline,
                        ),
                        _ => node.layer.propagate_ibp(node_input),
                    };
                    propagated.map_err(|e| {
                        if e.is_deadline_exceeded() {
                            e
                        } else {
                            NyError::InternalError(format!(
                                "Constraint forward: unary IBP failed at node '{}' ({}): {}",
                                node_name,
                                node.layer.layer_type(),
                                e
                            ))
                        }
                    })?
                };
                // Per-element intersection of re-propagated bounds with original CROWN-IBP
                // bounds to keep whichever is tighter in each dimension (#2935).
                repropagated
                    .intersection_per_element(current_bounds.as_ref())
                    .map(|(b, _)| b)
                    .unwrap_or(repropagated)
            };

            if let Some(cons) = lookups.pre.get(node_name) {
                output_bounds = apply_pre_constraints(&output_bounds, cons)?;
            }
            // #2399: Apply GenBaB pre-activation constraints (GeLU, Sigmoid, Tanh, etc.).
            // Tightens pre-activation bounds at arbitrary split points, analogous to how
            // ReLU pre-constraints tighten at 0.0.
            if let Some(cons) = lookups.pre_genbab.get(node_name) {
                output_bounds = apply_genbab_pre_constraints(&output_bounds, cons)?;
            }

            // In-cone (recomputed) entries are FRESH Arcs; out-of-cone entries
            // were never touched and stay `Arc`-shared with the parent's map.
            bounds_cache.insert(node_name.clone(), Arc::new(output_bounds));
        }

        ensure_constrained_propagation_deadline(
            deadline,
            "before publishing constrained forward bounds",
        )?;
        Ok((bounds_cache, constrained_input))
    }

    /// #cone-delta seed-source decision: return `Some(delta)` iff the delta
    /// seeds may replace the full-history seeds for the recompute cone.
    ///
    /// ALL of the following must hold (any failure ⇒ `None` ⇒ the caller keeps
    /// the full-history seeds, byte-identical to today):
    /// 1. `NY_CONE_REFRESH=1` (dark gate, default OFF).
    /// 2. A delta was supplied (`delta_seeds.is_some()`) and the caller passed
    ///    an inherited map (`base_bounds.is_some()`).
    /// 3. `clip_in_alpha_crown` is off and `NY_INTERM_REFINE`/`NY_STABILIZE`
    ///    are unset: each of those makes the final map a function of the WHOLE
    ///    history (clip_alpha.rs documents an explicit upstream-tightening
    ///    counterexample), so delta reuse would stay sound but break
    ///    byte-identity with the full-recompute reference — the gate
    ///    self-disables instead.
    /// 4. Every delta seed resolves in the graph, none is `NETWORK_INPUT`
    ///    (input splits change `input_bounds`, killing the idempotence premise
    ///    everywhere), and every one is among the full-history pre-activation
    ///    seeds — a delta entry outside the history's own pre-nodes means the
    ///    tracking is inconsistent with this history.
    /// 5. Every `exec_order` node has a `base_bounds` entry: with a partial
    ///    map, out-of-cone nodes have nothing to reuse.
    ///
    /// An EMPTY delta passing these checks is meaningful: no constraint was
    /// added since `base_bounds` was fixpointed, so the cone is empty and the
    /// whole inherited map is reused.
    fn select_delta_seeds<'d>(
        &self,
        graph: &GraphNetwork,
        delta_seeds: Option<&'d [String]>,
        full_seeds: &[String],
        exec_order: &[String],
        base_bounds: Option<NodeBoundsView<'_>>,
    ) -> Option<&'d [String]> {
        let delta = delta_seeds?;
        let base = base_bounds?;
        if !cone_refresh_enabled() {
            return None;
        }
        if self.config.clip_in_alpha_crown {
            return None;
        }
        if std::env::var("NY_INTERM_REFINE").is_ok() || std::env::var("NY_STABILIZE").is_ok() {
            return None;
        }
        let full_seed_set: std::collections::HashSet<&str> =
            full_seeds.iter().map(String::as_str).collect();
        if !delta.iter().all(|s| {
            s != NETWORK_INPUT
                && graph.nodes.contains_key(s.as_str())
                && full_seed_set.contains(s.as_str())
        }) {
            return None;
        }
        if !exec_order.iter().all(|n| base.contains_key(n.as_str())) {
            return None;
        }
        Some(delta)
    }

    /// CROWN propagation with intermediate storage for gradients.
    ///
    /// Same as `propagate_crown_with_graph_constraints` but stores A matrices at
    /// constrained ReLU nodes for analytical gradient computation. Delegates the
    /// backward pass to the shared core in `backward.rs`.
    pub(crate) fn propagate_crown_with_graph_constraints_storing_intermediates(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: Option<&GraphBetaState>,
        objective: Option<&[f32]>,
    ) -> Result<DomainCrownResultWithIntermediates> {
        let deadline = self.effective_graph_bab_deadline();
        ensure_constrained_propagation_deadline(
            deadline,
            "before constrained forward preparation",
        )?;
        let (mut bounds_cache, constrained_input, exec_order) =
            self.prepare_constrained_graph_bounds(graph, input, context, beta_state, objective)?;
        ensure_constrained_propagation_deadline(deadline, "before Complete Clip")?;
        self.maybe_apply_complete_clip_root_bank(
            graph,
            context,
            beta_state,
            objective,
            None,
            &constrained_input,
            &exec_order,
            &mut bounds_cache,
        );
        ensure_constrained_propagation_deadline(
            deadline,
            "after Complete Clip and before constrained backward preparation",
        )?;

        // Build constraint lookups for backward pass (needed to identify constrained ReLUs
        // for intermediate A-matrix storage).
        let lookups = build_constraint_lookups(
            &context.history.constraints,
            &context.history.genbab_constraints,
            graph,
        )?;

        ensure_constrained_propagation_deadline(deadline, "before constrained backward dispatch")?;
        // Delegate to shared backward CROWN core (storing intermediates mode).
        let params = BackwardParams {
            graph,
            constrained_input: &constrained_input,
            exec_order: &exec_order,
            context,
            beta_state,
            objective,
            spec_matrix: None,
            seed_cache: None,
            capture_linear_bounds: false,
            deadline, // #3795: thread BaB deadline
            patches_policy: ConstrainedPatchesPolicy::for_engine(context.engine),
        };
        let result = self.backward_crown_constrained(
            &params,
            &mut bounds_cache,
            BackwardMode::StoringIntermediates {
                lookups: Box::new(lookups),
            },
        )?;

        let intermediate = result.intermediate.ok_or_else(|| {
            NyError::InternalError(
                "StoringIntermediates mode did not produce intermediates".to_string(),
            )
        })?;

        Ok((result.output_bounds, bounds_cache, intermediate))
    }
}
