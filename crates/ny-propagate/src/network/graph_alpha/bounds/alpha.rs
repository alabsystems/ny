// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#[path = "alpha_reciprocal_support.rs"]
mod reciprocal_support;
#[path = "alpha_spec_objective.rs"]
mod spec_objective;
#[path = "alpha_sqrt_support.rs"]
mod sqrt_support;

use super::*;
use crate::layers::BoundPropagation;
use crate::network::alpha_crown_loop::finite_lower_sum;

/// Which collector actually produced an alpha reference-bounds map
/// (#dedup-root-collections Fix B).
///
/// The DAG alpha loop uses this to decide whether the init-collected map can
/// be reused for the pre-loop initial CROWN bound in place of the backward
/// pass's internal Step-1 collection: reuse only when the init map came from
/// the SAME collector Step 1 would run (or a strictly tighter one), so the
/// legacy per-graph-family bound quality is never weakened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AlphaReferenceBoundsSource {
    /// Per-node CROWN-IBP collection
    /// (`collect_crown_ibp_bounds_dag_with_status_and_deadline`).
    CrownIbp,
    /// Cached certified forward-linear substitution pass (conv DAGs).
    ForwardLinear,
    /// Plain IBP forward pass.
    Ibp,
}

impl GraphNetwork {
    /// Collect the intermediate bounds used to initialize alpha-CROWN state.
    ///
    /// With `fix_interm_bounds=true`, DAGs now follow the same IBP reference-bound
    /// contract as sequential graphs, except for the existing deep-sequential
    /// CROWN-IBP override (#3628, #4404).
    pub(crate) fn collect_alpha_reference_bounds_with_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn ny_core::GemmEngine>,
        exec_order: &[String],
    ) -> Result<std::collections::HashMap<String, BoundedTensor>> {
        self.collect_alpha_reference_bounds_with_engine_and_source(
            input, config, engine, exec_order,
        )
        .map(|(bounds, _source)| bounds)
    }

    /// Like [`Self::collect_alpha_reference_bounds_with_engine`] but also
    /// reports which collector produced the map (#dedup-root-collections).
    pub(crate) fn collect_alpha_reference_bounds_with_engine_and_source(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn ny_core::GemmEngine>,
        exec_order: &[String],
    ) -> Result<(
        std::collections::HashMap<String, BoundedTensor>,
        AlphaReferenceBoundsSource,
    )> {
        // #cgan-fwdlin-ref (DARK, `NY_FORWARD_LINEAR_CONV_TRANSPOSE_REF=1`):
        // cgan-class SEQUENTIAL conv chains (Gemm→Reshape→ConvTranspose→…)
        // never reach the conv-DAG forward-linear branch below (is_dag=false),
        // and the deep_seq CROWN-IBP early return additionally shadows it —
        // so the certified ConvTranspose/BatchNorm image surface was
        // unreachable exactly on the graphs it was built for. Under the dark
        // surface gate, try the certified forward-linear reference FIRST (a
        // chain is the easy case for forward substitution; the collector
        // requires no DAG-ness) and fall through to the shipped logic
        // UNCHANGED on any fail-closed refusal. Gate-off is byte-identical.
        // #cora-fwdlin-ref (DARK, `NY_FORWARD_LINEAR_SEQ_CONV_REF=1`): same
        // is_dag=false blocker, plain-Conv2d SEQUENTIAL chains (cora
        // cifar10-set: the reference collect burns ~14s of a 25s budget in the
        // single-threaded deep_seq CROWN-IBP sweep before α even starts).
        // Research gate for measuring forward-linear references on such
        // chains; gate-off is byte-identical.
        let seq_conv_gate = std::env::var("NY_FORWARD_LINEAR_SEQ_CONV_REF")
            .ok()
            .as_deref()
            == Some("1")
            && exec_order.iter().any(|n| {
                self.nodes
                    .get(n)
                    .is_some_and(|node| matches!(node.layer, Layer::Conv2d(_)))
            });
        if config.fix_interm_bounds
            && (seq_conv_gate
                || (GraphNetwork::forward_linear_conv_transpose_reference_enabled()
                    && exec_order.iter().any(|n| {
                        self.nodes
                            .get(n)
                            .is_some_and(|node| matches!(node.layer, Layer::ConvTranspose2d(_)))
                    })))
        {
            match self.collect_forward_linear_bounds_dag_cached(input, engine, config.deadline) {
                Ok(bounds) => {
                    info!(
                        "Forward-linear reference bounds (ConvTranspose surface, {} nodes)",
                        bounds.len()
                    );
                    return Ok(((*bounds).clone(), AlphaReferenceBoundsSource::ForwardLinear));
                }
                Err(
                    error @ (NyError::UnsupportedOp(_)
                    | NyError::UnsupportedConfiguration(_)
                    | NyError::DeadlineExceeded(_)
                    | NyError::ShapeMismatch { .. }
                    | NyError::CpuMemoryExceeded { .. }),
                ) => {
                    info!(
                        "Forward-linear ConvTranspose reference unavailable ({error}); \
                         falling through (fail-closed)"
                    );
                }
                Err(error) => return Err(error),
            }
        }
        let is_dag = !self.is_sequential_graph(exec_order);
        let act_count = exec_order
            .iter()
            .filter(|n| {
                self.nodes
                    .get(*n)
                    .is_some_and(|node| node.layer.requires_pre_activation_bounds())
            })
            .count();
        let deep_seq = !is_dag
            && config.fix_interm_bounds
            && act_count >= 3
            && self.should_use_crown_ibp_intermediates();

        // #linearizenn-dense-dag-ref: a DAG whose only DAG-ness is a tiny
        // skip path around an otherwise deep dense chain used to fall all the
        // way through to PLAIN IBP reference intermediates, which explode
        // through the chain. Measured on linearizenn_2024
        // AllInOne_120_120 / prop_120_120_0 (8x256 ReLU MLP plus a
        // Slice -> MatMul -> Concat input skip, 20 nodes):
        //   IBP intermediates:       root row bounds [-616.166, 473.220],
        //                            BaB best_gap -653.4 (never moves)
        //   CROWN-IBP intermediates: root row bounds [  36.070,  48.366],
        //                            BaB best_gap -5.32
        // i.e. a ~145x looser root objective purely from the reference
        // collector choice. The same three guards the sequential `deep_seq`
        // override already uses still apply (>=3 activations, no
        // binary-relaxation op, per-node CROWN-IBP node-count threshold), and
        // conv DAGs are excluded so the certified forward-linear reference
        // below keeps its image-scale surface. CROWN-IBP intermediates are
        // the same sound enclosures the `fix_interm_bounds=false` path
        // already ships, so this is a tightening, never a widening.
        //
        // KILL SWITCH (`NY_DENSE_DAG_REF=0`): a full 663-model structural scan of
        // the VNN-COMP 2025 set shows this arm changes the collector for THREE
        // categories, not the two the landing sample found — linearizenn_2024
        // (10/11 models), cersyve (36/36) and nn4sys `pensieve_small_simple`
        // (31 nodes, 12 acts, dense DAG; 10 instances currently all `unsat`).
        // The switch exists so the newly-reached categories can be A/B'd against
        // the pre-fix collector with ONE binary. Unset keeps the shipped arm.
        //
        // MEASURED with that switch, nn4sys `pensieve_small_simple` (the third
        // category, which the landing sample never touched), all 10 scored
        // instances at their official budgets, preset loaded:
        //   NY_DENSE_DAG_REF=0 (pre-fix collector)  10/10 unsat
        //   NY_DENSE_DAG_REF=1 (shipped arm)        10/10 unsat
        // No regression from the arm reaching a category nobody sampled.
        let dense_dag = is_dag
            && config.fix_interm_bounds
            && act_count >= 3
            && !self.has_conv_layers()
            && self.should_collect_per_node_crown_ibp_intermediates()
            && !matches!(std::env::var("NY_DENSE_DAG_REF").ok().as_deref(), Some("0"));

        if !config.fix_interm_bounds || deep_seq || dense_dag {
            info!(
                "CROWN-IBP reference bounds (dag={is_dag}, deep_seq={deep_seq}, \
                 dense_dag={dense_dag}, {act_count} acts)"
            );
            return Ok((
                self.collect_crown_ibp_bounds_dag_with_status_and_deadline(
                    input,
                    config.deadline,
                    engine,
                )?
                .bounds,
                AlphaReferenceBoundsSource::CrownIbp,
            ));
        }

        // Conv-DAG forward-linear reference bounds (#vnncomp-image-forward-linear):
        // plain IBP explodes through deep conv ResNets (cifar100: 17 convs) until
        // the CROWN backward NaN firewall fires and the root bound is vacuous.
        // The certified forward-substitution pass gives O(L) finite intermediates.
        // Default ON for conv DAGs; disable with NY_NO_FORWARD_LINEAR_REF=1
        // (disable-flag, never enable-flag). Fails closed to plain IBP on any
        // unsupported op / deadline / memory-cap refusal.
        let conv_dag = is_dag
            && exec_order.iter().any(|n| {
                self.nodes
                    .get(n)
                    .is_some_and(|node| matches!(node.layer, Layer::Conv2d(_)))
            });
        if conv_dag && GraphNetwork::forward_linear_reference_enabled() {
            match self.collect_forward_linear_bounds_dag_cached(input, engine, config.deadline) {
                Ok(bounds) => {
                    info!(
                        "Forward-linear reference bounds (conv DAG, {} nodes, fix_interm_bounds=true)",
                        bounds.len()
                    );
                    return Ok(((*bounds).clone(), AlphaReferenceBoundsSource::ForwardLinear));
                }
                Err(
                    error @ (NyError::UnsupportedOp(_)
                    | NyError::UnsupportedConfiguration(_)
                    | NyError::DeadlineExceeded(_)
                    | NyError::ShapeMismatch { .. }
                    | NyError::CpuMemoryExceeded { .. }),
                ) => {
                    info!(
                        "Forward-linear reference bounds unavailable ({error}); \
                         falling back to plain IBP (fail-closed)"
                    );
                }
                Err(error) => return Err(error),
            }
        }

        info!("IBP reference bounds (dag={is_dag}, deep_seq={deep_seq}, fix_interm_bounds=true)");
        Ok((
            self.collect_node_bounds_with_engine_and_deadline(input, engine, config.deadline)?,
            AlphaReferenceBoundsSource::Ibp,
        ))
    }

    /// Collect α-CROWN bounds plus reusable graph alpha state.
    pub fn collect_alpha_crown_bounds_dag(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
    ) -> Result<GraphAlphaCollectionResult> {
        self.collect_alpha_crown_bounds_dag_with_engine(input, config, None)
    }

    /// Collect α-CROWN bounds for DAG models with optional GEMM acceleration.
    pub fn collect_alpha_crown_bounds_dag_with_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn ny_core::GemmEngine>,
    ) -> Result<GraphAlphaCollectionResult> {
        // Disable the L2/Cauchy–Schwarz lever for the DAG alpha-CROWN scope. This
        // entry is reached both from `propagate_alpha_crown_with_config_and_engine_impl`
        // (via the DAG delegate) and directly from beta-CROWN root-bound collection;
        // the guard nests harmlessly when an outer CROWN guard is already active.
        // Covers the per-iteration / per-node IBP reference + intermediate bound
        // recomputation. Sound (lever only tightens); restored on drop.
        let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
        let exec_order = self.exec_order()?;
        if let Some(result) = self.maybe_collect_sequential_alpha_crown_bounds_with_engine(
            exec_order, input, config, engine,
        ) {
            return result;
        }

        // #4036: non-SPSA gradient methods delegate to the DAG optimizer
        // which dispatches AnalyticChain, FD, and Analytic gradients.
        if let Some(result) = self.try_dag_gradient_dispatch(input, config, engine, exec_order)? {
            return Ok(result);
        }

        // Step 1: collect bounds at all nodes. (#3357)
        let reference_bounds =
            self.collect_alpha_reference_bounds_with_engine(input, config, engine, exec_order)?;
        // Step 2: Initialize graph alpha state for all optimizable activation nodes.
        let mut alpha_state = GraphAlphaState::new();
        let relu_nodes: Vec<String> = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::ReLU(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        for relu_name in &relu_nodes {
            let pre_act = self.relu_preactivation_bounds(
                relu_name,
                input,
                &reference_bounds,
                "collect-alpha-crown-dag-init",
            )?;
            alpha_state.add_relu_node(relu_name, pre_act, !config.full_conv_alpha)?;
        }
        let s_shaped_nodes: Vec<String> = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::Sigmoid(_) | Layer::Tanh(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        for node_name in &s_shaped_nodes {
            let node = self.nodes.get(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("S-shaped node {} not found", node_name))
            })?;
            let pre_activation = reference_bounds.get(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Pre-activation bounds for monotone node '{}' not found",
                    node_name
                ))
            })?;
            match &node.layer {
                Layer::Sigmoid(_) => alpha_state.add_sigmoid_node(node_name, pre_activation)?,
                Layer::Tanh(_) => alpha_state.add_tanh_node(node_name, pre_activation)?,
                _ => {}
            }
        }
        let sqrt_nodes = sqrt_support::initialize_sqrt_alpha_nodes(
            self,
            exec_order,
            input,
            &reference_bounds,
            &mut alpha_state,
        )?;
        let reciprocal_nodes = reciprocal_support::initialize_reciprocal_alpha_nodes(
            self,
            exec_order,
            input,
            &reference_bounds,
            &mut alpha_state,
        )?;
        let num_unstable = alpha_state.num_unstable();
        let num_monotone = alpha_state.monotone_alpha_names().count();
        let num_sqrt = alpha_state.sqrt_alpha_names().count();
        let num_reciprocal = alpha_state.reciprocal_alpha_names().count();
        if num_unstable == 0 && num_monotone == 0 && num_sqrt == 0 && num_reciprocal == 0 {
            debug!("GraphNetwork α-CROWN: No optimizable activation state, using IBP bounds");
            return Ok((reference_bounds, alpha_state));
        }

        // Adaptive skip: for deep models, alpha-CROWN optimization provides
        // diminishing returns due to bound explosion through many ReLU layers.
        // Skip optimization and return IBP/CROWN-IBP bounds directly. This
        // matches the check in propagate_dag.rs and saves ~20 backward passes
        // for 16+ layer models like malbeware 16-25. #3218
        if config.adaptive_skip && relu_nodes.len() > config.adaptive_skip_depth_threshold {
            info!(
                "GraphNetwork α-CROWN: adaptive skip — {} ReLU nodes > threshold {}, returning bounds without optimization",
                relu_nodes.len(),
                config.adaptive_skip_depth_threshold
            );
            return Ok((reference_bounds, alpha_state));
        }

        debug!(
            "GraphNetwork α-CROWN: Starting optimization with {} unstable ReLU neurons, {} monotone nodes, {} sqrt nodes, {} ReLU nodes, {} iterations",
            num_unstable,
            num_monotone,
            num_sqrt,
            relu_nodes.len(),
            config.iterations
        );

        // Step 3: Optimization loop — maximize output lower bound (matches α,β-CROWN).
        let output_node = if self.output_node.is_empty() {
            exec_order.last().cloned().ok_or_else(|| {
                NyError::InvalidSpec(
                    "α-CROWN optimization: output_node is empty and exec_order is empty — no node to optimize".to_string()
                )
            })?
        } else {
            self.output_node.clone()
        };

        let mut best_bounds: std::collections::HashMap<String, BoundedTensor> =
            reference_bounds.clone();
        let mut best_output_lower = f32::NEG_INFINITY;
        // Plateau early-exit (early_stop_patience): track consecutive iterations
        // with no meaningful improvement to best_output_lower, mirroring the DAG
        // loop in propagate_dag/mod.rs:278-299. `prev_best_lower_sum` snapshots the
        // best at the START of each iteration so `best_improvement` measures the gain
        // made by THIS iteration. Plateau-exit only stops optimizing sooner; it
        // returns the best valid bound already found (no bound computation changes).
        let mut prev_best_lower_sum = best_output_lower;
        let mut no_improve_iters = 0usize;
        // Element-wise best output bounds: track per-dimension tightest lower/upper
        // across iterations. Without this, a scalar sum comparison can discard
        // per-dimension tightness improvements. See #2251.
        let mut best_output_lower_arr: Option<ArrayD<f32>> = None;
        let mut best_output_upper_arr: Option<ArrayD<f32>> = None;
        let mut lr = config.learning_rate;
        let eps = 1e-3; // Perturbation magnitude for SPSA

        // Sparse optimization: track which alphas are "active" (being optimized)
        // After first iteration, keep only top sparse_ratio fraction by gradient magnitude
        let mut sparse_mask: Option<std::collections::BTreeMap<String, Array1<bool>>> = None;
        let use_sparse = config.sparse_ratio < 1.0 && config.sparse_ratio > 0.0;

        for iter in 0..config.iterations {
            // Deadline check (#2698): bail early if verification timeout budget
            // is exhausted. Return current best bounds instead of running all iterations.
            if config.past_deadline() {
                info!(
                    "GraphNetwork α-CROWN: deadline exceeded at iteration {}/{}, returning best bounds",
                    iter, config.iterations
                );
                break;
            }

            // Compute CROWN bounds at OUTPUT node only (for efficiency)
            // We only need output bounds for the objective during optimization.
            // #3218: Catch UnsupportedOp/UnsupportedConfiguration from nodes in the
            // backward path (e.g., Gather) and fall back to IBP bounds. The
            // best_bounds is initialized to reference_bounds, so breaking out returns
            // correct (conservative) IBP bounds. Matches the catch at line 423 for
            // intermediate bounds in compute_all_crown_bounds_with_alpha().
            let output_bounds = match self.propagate_crown_to_node_with_alpha(
                input,
                &output_node,
                &std::collections::HashMap::new(), // Don't need intermediate CROWN bounds
                &reference_bounds,
                &alpha_state,
                engine,
                config.deadline,
            ) {
                Ok(bounds) => bounds,
                // CpuMemoryExceeded is the Conv2d backward memory-cap backstop
                // (#conv-crown-oom); IBP is the sound fallback, same as the others.
                Err(
                    NyError::UnsupportedOp(_)
                    | NyError::UnsupportedConfiguration(_)
                    | NyError::ShapeMismatch { .. }
                    | NyError::CpuMemoryExceeded { .. }
                    | NyError::DeadlineExceeded(_),
                ) => {
                    info!(
                        "α-CROWN: backward to '{}' failed, returning IBP (#3602)",
                        output_node
                    );
                    break;
                }
                Err(e) => return Err(e),
            };

            // Intersect the raw backward output bound with the always-available IBP output
            // bound before it drives the objective / elementwise-best / spec early-exit.
            // SOUND: both enclose the output node's reachable set, so the per-element
            // intersection (union on disjoint) still encloses it — never looser; on
            // NaN/shape-mismatch (None) keep the CROWN bound unchanged (sound; NaN-guarded
            // downstream). Mirrors the post-loop IBP intersection, moved earlier.
            let output_bounds = match reference_bounds.get(&output_node) {
                Some(ibp_out) if ibp_out.shape() == output_bounds.shape() => output_bounds
                    .intersection_per_element(ibp_out)
                    .map(|(t, _)| t)
                    .unwrap_or(output_bounds),
                _ => output_bounds,
            };

            // Compute objective: sum of OUTPUT lower bounds (higher is better).
            // Filter -Inf elements to prevent NaN from (-Inf)-(-Inf) in
            // early-stopping and gradient computation (#3272, #2857).
            let output_lower: f32 = finite_lower_sum(output_bounds.lower());

            // NaN guard (#2663): If upstream instability produces NaN bounds,
            // skip bounds update to prevent NaN from contaminating best bounds
            // and downstream CROWN backward passes. LR decay still occurs.
            let output_nan = !output_lower.is_finite()
                || output_bounds.lower().iter().any(|v| v.is_nan())
                || output_bounds.upper().iter().any(|v| v.is_nan());
            if output_nan {
                warn!(
                    iter = iter,
                    output_lower = output_lower,
                    "α-CROWN: NaN/Inf in output bounds, skipping bounds update",
                );
            } else {
                // Update scalar best for logging
                if output_lower > best_output_lower {
                    best_output_lower = output_lower;
                    debug!(
                        "GraphNetwork α-CROWN: iter {} improved output lower to {:.4}",
                        iter, output_lower
                    );
                }

                // Update element-wise best output bounds (#2251).
                // Matches the sequential alpha-CROWN pattern in alpha_crown.rs and
                // propagate_dag.rs which call update_elementwise_best_bounds().
                // Skip updates during warmup window to avoid locking in noisy
                // early-iteration bounds (start_save_best, optimized_bounds.py:785-797).
                let is_last_iter = iter == config.iterations - 1;
                match (&mut best_output_lower_arr, &mut best_output_upper_arr) {
                    (Some(ref mut best_l), Some(ref mut best_u)) => {
                        if config.should_save_best(iter, is_last_iter) {
                            crate::network::graph_alpha::propagate_helpers::update_elementwise_best_bounds(
                                best_l,
                                best_u,
                                &output_bounds,
                                iter,
                            )?;
                        }
                    }
                    _ => {
                        // First iteration: initialize from current output bounds
                        best_output_lower_arr = Some(output_bounds.lower().clone());
                        best_output_upper_arr = Some(output_bounds.upper().clone());
                    }
                }

                // Spec-proven early-exit (#warmup-early-exit). When the single-objective
                // warmup carries a `spec_early_exit`, project the elementwise BEST output
                // bounds (the bounds this loop will actually return) onto the objective and
                // stop the moment they already prove the property against the threshold.
                // We project the best (not just the current iteration) so the bound we exit
                // on is exactly the one returned. SOUND: this only stops optimizing sooner —
                // the projected bound at the exit iteration is a valid over-approximation
                // that clears the threshold (proving the property); no bound *computation*
                // changes, and `None` callers never reach this branch.
                if let Some(spec) = config.spec_early_exit.as_ref() {
                    let (proj_lo, proj_hi) = match (&best_output_lower_arr, &best_output_upper_arr)
                    {
                        (Some(best_l), Some(best_u)) => (
                            best_l.as_slice().map(|s| s.to_vec()),
                            best_u.as_slice().map(|s| s.to_vec()),
                        ),
                        _ => (
                            output_bounds.lower().as_slice().map(|s| s.to_vec()),
                            output_bounds.upper().as_slice().map(|s| s.to_vec()),
                        ),
                    };
                    if let (Some(lo), Some(hi)) = (proj_lo, proj_hi) {
                        if let Some((root_lower, root_upper)) = spec.project_bounds(&lo, &hi) {
                            if spec.is_verified(root_lower, root_upper) {
                                debug!(
                                    "GraphNetwork α-CROWN: spec-proven early-exit at iter {} \
                                     (root bound [{:.4}, {:.4}] clears threshold {})",
                                    iter, root_lower, root_upper, spec.threshold
                                );
                                break;
                            }
                        }
                    }
                }
            }

            // Plateau early-exit (early_stop_patience). Mirrors the DAG loop in
            // propagate_dag/mod.rs:278-299 EXACTLY: measure the improvement this
            // iteration made to the scalar best, count consecutive non-improving
            // iterations against `config.tolerance`, and stop once we've plateaued
            // for `early_stop_patience` iterations. SOUND: this only stops the
            // optimization loop sooner — `best_output_lower_arr`/`best_output_upper_arr`
            // already hold the tightest valid (over-approximate) bounds seen so far,
            // and nothing about bound computation changes.
            let best_improvement = best_output_lower - prev_best_lower_sum;
            if best_improvement < config.tolerance {
                no_improve_iters += 1;
            } else {
                no_improve_iters = 0;
            }
            if iter > 0 && no_improve_iters >= config.early_stop_patience {
                debug!(
                    "GraphNetwork α-CROWN: Converged at iteration {} (best improvement < {} for {} iters)",
                    iter, config.tolerance, no_improve_iters
                );
                break;
            }
            prev_best_lower_sum = best_output_lower;

            // Skip gradient update on last iteration
            if iter == config.iterations - 1 {
                break;
            }

            // Compute gradients using SPSA - targeting output objective
            // Pass sparse_mask to only perturb active alphas (reduces SPSA variance)
            let gradients = self.compute_spsa_gradients_dag_for_output_sparse(
                input,
                &reference_bounds,
                &alpha_state,
                &output_node,
                eps,
                config.spsa_samples,
                sparse_mask.as_ref(),
                engine,
            )?;

            // After first iteration, select top alphas by gradient magnitude.
            // Skipped when gradients are NaN because NaN magnitudes would produce
            // a garbage mask. If iter 0 has NaN, sparse mode won't activate —
            // acceptable since sparse is an optimization, not a correctness requirement.
            // Non-finite gradient check is only needed for this sparse mask decision;
            // per-ReLU guards in the update loop below handle individual NaN gradients.
            if iter == 0
                && use_sparse
                && !gradients
                    .relu
                    .values()
                    .any(|g| g.iter().any(|v| !v.is_finite()))
            {
                sparse_mask = Some(Self::select_top_alphas(
                    &gradients.relu,
                    config.sparse_ratio,
                ));
                let active_count: usize = sparse_mask
                    .as_ref()
                    .map(|m| m.values().map(|v| v.iter().filter(|&&b| b).count()).sum())
                    .unwrap_or(0);
                debug!(
                    "GraphNetwork α-CROWN: Sparse mode enabled, optimizing top {} alphas ({}% of {})",
                    active_count,
                    (config.sparse_ratio * 100.0) as usize,
                    num_unstable
                );
            }

            // Update alpha values with gradient ascent (maximize output lower bound).
            // Per-ReLU non-finite guard (#2867, #2835): only skip the affected ReLU,
            // not all ReLUs. Matches the per-ReLU guard pattern in
            // alpha_crown_loop.rs:188 and propagate_dag.rs:670.
            let adam_params = config.adam_params(lr, iter + 1);
            for relu_name in &relu_nodes {
                if let Some(grad) = gradients.relu.get(relu_name) {
                    // Per-ReLU non-finite guard: skip only this ReLU if its gradient
                    // contains NaN/Inf. Healthy ReLUs still get updated. (#2867)
                    if grad.iter().any(|v| !v.is_finite()) {
                        warn!(
                            iter = iter,
                            relu = relu_name.as_str(),
                            "α-CROWN: non-finite gradient for {relu_name}, skipping update (#2867)",
                        );
                        continue;
                    }
                    let mask = sparse_mask.as_ref().and_then(|m| m.get(relu_name));
                    // Negate because we want to maximize, but update() does gradient descent
                    let neg_grad = if let Some(mask_arr) = mask {
                        // Zero out gradients for inactive alphas
                        let masked: Vec<f32> = grad
                            .iter()
                            .zip(mask_arr.iter())
                            .map(|(&g, &active)| if active { -g } else { 0.0 })
                            .collect();
                        Array1::from_vec(masked)
                    } else {
                        -grad
                    };
                    // Channel-only alpha reduction (#4404): reduce per-neuron gradient
                    // to per-channel when full_conv_alpha is False.
                    let neg_grad = alpha_state.reduce_gradient(relu_name, &neg_grad);
                    // Update lower + upper alpha paths (#3782, cf. propagate_dag.rs:1597).
                    match config.optimizer {
                        Optimizer::Adam => {
                            alpha_state.update_adam(relu_name, &neg_grad, &adam_params);
                            alpha_state.update_adam_upper(relu_name, &neg_grad, &adam_params);
                        }
                        Optimizer::Sgd => {
                            alpha_state.update(relu_name, &neg_grad, lr, config.momentum);
                            alpha_state.update_upper(relu_name, &neg_grad, lr, config.momentum);
                        }
                    }
                }
            }
            for node_name in &s_shaped_nodes {
                if let Some(grad) = gradients.monotone.get(node_name) {
                    if grad.any_non_finite() {
                        warn!(
                            iter = iter,
                            node = node_name.as_str(),
                            "α-CROWN: non-finite monotone gradient for {node_name}, skipping update (#3619)"
                        );
                        continue;
                    }
                    let neg_grad = grad.negate();
                    if let Some(alpha) = alpha_state.monotone_s_shaped_alpha_mut(node_name) {
                        match config.optimizer {
                            Optimizer::Adam => alpha.update_adam(&neg_grad, &adam_params),
                            Optimizer::Sgd => alpha.update_sgd(&neg_grad, lr, config.momentum),
                        }
                    }
                }
            }
            sqrt_support::update_sqrt_alpha_gradients(
                &sqrt_nodes,
                &gradients.sqrt,
                &mut alpha_state,
                config.optimizer,
                &adam_params,
                lr,
                config.momentum,
                iter,
            );
            reciprocal_support::update_reciprocal_alpha_gradients(
                &reciprocal_nodes,
                &gradients.reciprocal,
                &mut alpha_state,
                config.optimizer,
                &adam_params,
                lr,
                config.momentum,
                iter,
            );

            // Learning rate decay — always applied to keep schedule consistent
            // even when NaN causes gradient or bounds updates to be skipped (#2663).
            lr *= config.lr_decay;
        }

        // When fix_interm_bounds=true, skip the expensive O(N²) post-optimization
        // CROWN computation and return the Step 1 reference bounds directly.
        // When fix_interm_bounds=false, do the full alpha-CROWN pass.
        if config.fix_interm_bounds {
            debug!("GraphNetwork α-CROWN: Using Step 1 reference bounds (skipping post-opt CROWN)");
            best_bounds = reference_bounds;
        } else {
            // After optimization, compute full intermediate bounds with optimized alphas
            let current_bounds = self.collect_crown_bounds_with_alpha(
                input,
                &reference_bounds,
                &alpha_state,
                engine,
                config.deadline,
            )?;

            // Intersect with IBP for soundness
            for (name, reference_bound) in &reference_bounds {
                if let Some(crown_bound) = current_bounds.get(name) {
                    if crown_bound.shape() == reference_bound.shape() {
                        // Per-element intersection with union fallback (#2935).
                        let (tightened, disjoint) = reference_bound
                            .intersection_per_element(crown_bound)
                            .unwrap_or_else(|| {
                                debug!(
                                    "α-CROWN: {} IBP/CROWN intersection failed (NaN/shape), using reference bounds",
                                    name
                                );
                                (reference_bound.clone(), 0)
                            });
                        if disjoint > 0 {
                            debug!(
                                "α-CROWN: {} IBP/CROWN intersection: {} of {} elements disjoint, used union fallback",
                                name, disjoint, tightened.len()
                            );
                        }
                        best_bounds.insert(name.clone(), tightened);
                    } else {
                        debug!(
                            "α-CROWN: {} shape mismatch reference={:?} vs CROWN={:?}, using reference bounds",
                            name,
                            reference_bound.shape(),
                            crown_bound.shape()
                        );
                    }
                }
            }
        }

        // Merge element-wise best output bounds into the output node's bounds (#2251).
        // The optimization loop tracked per-dimension best lower/upper across iterations.
        // Without this, the scalar sum comparison can discard per-dimension improvements.
        if let (Some(mut ew_lower), Some(mut ew_upper)) =
            (best_output_lower_arr, best_output_upper_arr)
        {
            // Widen inversions that arise from cross-iteration elementwise merge
            // (same pattern as propagate_dag.rs and propagate_sequential.rs).
            crate::network::graph_alpha::propagate_helpers::clamp_inverted_best_bounds(
                &mut ew_lower,
                &mut ew_upper,
                "collect_alpha_crown_bounds_dag output",
            );

            match BoundedTensor::new_allow_infinite(ew_lower, ew_upper) {
                Ok(ew_bounds) => {
                    // Intersect element-wise best with existing output bounds for tightening.
                    if let Some(existing) = best_bounds.get(&output_node) {
                        if existing.shape() == ew_bounds.shape() {
                            let (tightened, disjoint) =
                                existing.intersection_per_element(&ew_bounds).unwrap_or_else(|| {
                                    debug!(
                                        "α-CROWN: elementwise best / existing intersection failed (NaN/shape), using existing"
                                    );
                                    (existing.clone(), 0)
                                });
                            if disjoint > 0 {
                                debug!(
                                    "α-CROWN: elementwise best / existing: {} of {} elements disjoint, used union fallback",
                                    disjoint, tightened.len()
                                );
                            }
                            best_bounds.insert(output_node.clone(), tightened);
                        }
                    } else {
                        best_bounds.insert(output_node.clone(), ew_bounds);
                    }
                }
                Err(e) => {
                    warn!(
                        "α-CROWN: BoundedTensor::new_allow_infinite failed for output node {:?} — \
                         per-dimension bound improvement dropped: {e}",
                        output_node
                    );
                }
            }
        }
        debug!("GraphNetwork α-CROWN: Finished optimization, final output_lower={best_output_lower:.4}");
        Ok((best_bounds, alpha_state))
    }
}
