// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DAG α-CROWN propagation for `GraphNetwork`.
//!
//! Contains the α-CROWN implementation for non-sequential (DAG) graphs like ResNet
//! with skip connections and multiple paths.

mod alpha_update;
mod collect;
mod diagnostics;
mod gradients;
mod init;
mod supplements;

use init::{DagAlphaInitResult, DagAlphaInitState};

use crate::bounds::AlphaCrownConfig;

use crate::bounds::GraphAlphaState;
use ndarray::{Array1, ArrayD};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info, instrument, warn};

use super::propagate_helpers::{
    bounds_infeasible, clamp_inverted_best_bounds, update_elementwise_best_bounds,
};
use super::resnet_decompose::root_alpha_gpu_enabled;
use super::resnet_skeleton::{build_resnet_segment_skeleton, extract_skeleton_enabled};
use crate::network::alpha_crown_loop::{
    alpha_iteration_needs_gradient, final_alpha_bound_only_enabled, finite_lower_sum,
};
use crate::network::core::GraphNetwork;
use crate::network::graph_alpha::bounds::AlphaReferenceBoundsSource;

const DEFAULT_ALPHA_REFRESH_FRACTION: f32 = 0.25;
const MIN_ALPHA_REFRESH_FRACTION: f32 = 0.01;

/// Whether the caller consumes the post-loop alpha/optimizer state.
///
/// The final gradient/update is dead only for [`Self::BoundsOnly`]. Collection
/// returns the state as a BaB warm start and immediately re-evaluates it, so
/// skipping that update would change an observable result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DagAlphaLoopResultUse {
    BoundsOnly,
    BoundsAndState,
}

impl DagAlphaLoopResultUse {
    fn terminal_bound_only(self, gate_enabled: bool) -> bool {
        gate_enabled && matches!(self, Self::BoundsOnly)
    }
}

/// Parse the share of the remaining root budget available to the complete
/// sequence of intermediate alpha-bound refreshes. Invalid or out-of-range
/// values preserve the shipped default so a malformed measurement override
/// cannot silently disable or monopolize the refresh lane.
fn parse_alpha_refresh_fraction(raw: Option<&str>) -> f32 {
    raw.and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|fraction| (MIN_ALPHA_REFRESH_FRACTION..=1.0).contains(fraction))
        .unwrap_or(DEFAULT_ALPHA_REFRESH_FRACTION)
}

fn alpha_refresh_fraction_from_env() -> f32 {
    let raw = std::env::var("NY_ALPHA_REFRESH_FRACTION").ok();
    parse_alpha_refresh_fraction(raw.as_deref())
}

/// Lazily create one cumulative refresh pool and return the allowance for the
/// next refresh.
///
/// The global remainder can shrink independently, so every allowance is
/// clamped to both limits. The fraction is applied exactly once: subsequent
/// improving iterations only receive what remains in the original pool.
fn cumulative_alpha_refresh_allowance(
    budget_remaining: &mut Option<std::time::Duration>,
    global_remaining: std::time::Duration,
    fraction: f32,
) -> std::time::Duration {
    let budget = budget_remaining.get_or_insert_with(|| global_remaining.mul_f32(fraction));
    (*budget).min(global_remaining)
}

/// Charge actual refresh airtime to the cumulative pool.
///
/// A collector can finish just after its local deadline. Saturating subtraction
/// makes that overrun exhaust the pool instead of accidentally creating more
/// refresh time.
fn debit_alpha_refresh_budget(
    budget_remaining: &mut Option<std::time::Duration>,
    elapsed: std::time::Duration,
) {
    if let Some(budget) = budget_remaining.as_mut() {
        *budget = budget.saturating_sub(elapsed);
    }
}

/// Reuse init-collected bounds only when they match (or tighten) the collector
/// graph-CROWN Step 1 would independently select.
fn can_reuse_initial_node_bounds(
    source: AlphaReferenceBoundsSource,
    step1_would_use_forward_linear: bool,
) -> bool {
    match source {
        AlphaReferenceBoundsSource::CrownIbp => !step1_would_use_forward_linear,
        AlphaReferenceBoundsSource::ForwardLinear => step1_would_use_forward_linear,
        AlphaReferenceBoundsSource::Ibp => false,
    }
}

/// Immutable context for the DAG alpha optimization loop.
///
/// Bundles parameters that are fixed for the entire optimization. Mutable state
/// (`runtime`, `bilinear_alphas`, etc.) is passed as separate `&mut` parameters
/// so the borrow checker can reason about independent borrows.
pub(super) struct DagAlphaLoopContext<'a> {
    pub(super) input: &'a BoundedTensor,
    pub(super) exec_order: &'a [String],
    pub(super) output_dim: usize,
    pub(super) input_dim: usize,
    pub(super) config: &'a AlphaCrownConfig,
    pub(super) engine: Option<&'a dyn GemmEngine>,
    pub(super) relu_nodes: &'a [(String, usize)],
    pub(super) has_bilinear: bool,
    pub(super) has_mul_binary: bool,
}

impl GraphNetwork {
    /// α-CROWN for DAG (non-sequential) graphs like ResNet with optional GEMM acceleration.
    ///
    /// This handles graphs with skip connections (Add operations) and multiple paths.
    /// The backward pass accumulates linear bounds from all consumers of each node.
    #[instrument(skip(self, input, config, engine), fields(num_nodes = self.nodes.len(), iterations = config.iterations))]
    pub(super) fn propagate_dag_alpha_crown_with_config_and_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let init = match self.init_dag_alpha_state(input, config, engine)? {
            DagAlphaInitResult::EarlyReturn(bounds) => return Ok(bounds),
            DagAlphaInitResult::Ready(state) => *state,
        };
        self.dag_alpha_optimize_loop(
            input,
            config,
            engine,
            init,
            final_alpha_bound_only_enabled(),
            DagAlphaLoopResultUse::BoundsOnly,
        )
        .map(|(bounds, _alpha_state)| bounds)
    }

    /// Shared optimization loop for DAG α-CROWN.
    ///
    /// Returns the optimized output bounds and the final alpha state.
    fn dag_alpha_optimize_loop(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        init: DagAlphaInitState,
        final_bound_only_gate: bool,
        result_use: DagAlphaLoopResultUse,
    ) -> Result<(BoundedTensor, GraphAlphaState)> {
        let DagAlphaInitState {
            node_bounds,
            node_bounds_source,
            exec_order,
            output_dim,
            input_dim,
            relu_nodes,
            mut runtime,
            mut bilinear_alphas,
            mut bilinear_adam_m,
            mut bilinear_adam_v,
            mut mul_binary_alphas,
            mut mul_binary_adam_m,
            mut mul_binary_adam_v,
            has_bilinear,
            has_mul_binary,
            has_s_shaped,
            has_sqrt,
            has_reciprocal,
            invprop_enabled,
        } = init;

        let num_unstable = runtime.graph().num_unstable();
        let num_s_shaped = runtime.graph().monotone_alpha_names().count();
        let num_sqrt = runtime.graph().sqrt_alpha_names().count();
        debug!(
            "DAG α-CROWN: Starting optimization with {} unstable ReLU neurons across {} ReLU nodes, {} monotone S-shaped nodes, and {} sqrt nodes{}",
            num_unstable,
            relu_nodes.len(),
            num_s_shaped,
            num_sqrt,
            if invprop_enabled {
                " (INVPROP enabled)"
            } else {
                ""
            }
        );

        // Adaptive skip: check if network is too deep for α-CROWN to help
        if config.adaptive_skip && relu_nodes.len() > config.adaptive_skip_depth_threshold {
            info!(
                "DAG α-CROWN: Adaptive skip triggered - {} ReLU nodes > threshold {}. \
                 For deep networks, bounds are often fundamentally loose and α-CROWN optimization \
                 provides no benefit. Falling back to CROWN.",
                relu_nodes.len(),
                config.adaptive_skip_depth_threshold
            );
            let alpha_state = runtime.into_graph_alpha_state();
            return self
                .propagate_crown_with_engine_and_deadline(input, engine, config.deadline)
                .map(|r| (r.bounds, alpha_state));
        }

        if tracing::enabled!(tracing::Level::DEBUG) {
            self.log_pre_loop_diagnostics(&exec_order, &node_bounds, &relu_nodes, input)?;
        }

        let ctx = DagAlphaLoopContext {
            input,
            exec_order: &exec_order,
            output_dim,
            input_dim,
            config,
            engine,
            relu_nodes: &relu_nodes,
            has_bilinear,
            has_mul_binary,
        };

        // #root-alpha-gpu (A): build the warmup segment skeleton ONCE per loop
        // so every per-iteration warmup fold (bound + gradients) re-bakes only
        // the per-domain slots — static `Arc` weight payloads stay shared
        // across iterations instead of being re-materialized each extraction.
        // Dark behind NY_ROOT_ALPHA_GPU=1 (default OFF ⇒ the field stays
        // `None` and every warmup site takes the legacy extraction,
        // byte-identically). A build refusal also leaves `None` — fail closed.
        // `allow_pure_chain=false` matches the warmup extraction sites exactly.
        if root_alpha_gpu_enabled() && extract_skeleton_enabled() {
            let output_node_name = if self.output_node.is_empty() {
                exec_order.last().cloned()
            } else {
                Some(self.output_node.clone())
            };
            if let Some(output_node) = output_node_name {
                let skeleton = build_resnet_segment_skeleton(
                    self,
                    input,
                    &output_node,
                    &node_bounds,
                    &node_bounds,
                    Some(runtime.graph()),
                    /*allow_pure_chain=*/ false,
                );
                runtime.set_warmup_skeleton(skeleton);
            }
        }

        // #w4-gpu-dag-backward: the pre-loop initial CROWN bound below was measured
        // (sample profile, cifar100 CIFAR100_resnet_medium) at ~40s CPU wall — the
        // ENTIRE warmup budget, so alpha finished 0/20 iterations. Route it through
        // the SOUND GPU-resident resnet backward (identity seed, CROWN-initialized
        // alpha folded — the same certified enclosure the in-loop warmup bound uses)
        // when the suffix decomposes; on any refusal the proven CPU path below runs
        // unchanged. The expected output shape is captured BEFORE `node_bounds`
        // moves into `reference_bounds` so the flat GPU bound can be reshaped to
        // match the CPU path's output layout.
        let gpu_initial_bounds = self.try_gpu_warmup_bound(&ctx, &node_bounds, &runtime);
        let output_shape: Option<Vec<usize>> = gpu_initial_bounds.as_ref().and_then(|_| {
            let output_node_name = if self.output_node.is_empty() {
                exec_order.last()?
            } else {
                &self.output_node
            };
            node_bounds
                .get(output_node_name)
                .map(|b| b.shape().to_vec())
        });

        let mut reference_bounds = super::reference_bounds::GraphAlphaReferenceBounds::new(
            node_bounds,
            self.graph_alpha_reference_bound_targets()?,
        )?;

        // Step 3: Optimization loop
        // Track element-wise best bounds across iterations:
        // - best_lower: maximum lower bound seen for each output dimension
        // - best_upper: minimum upper bound seen for each output dimension
        // Initialize from CROWN bounds to ensure α-CROWN never returns worse bounds.
        let crown_bounds = match gpu_initial_bounds {
            Some(bounds) => {
                info!(
                    "DAG α-CROWN: initial CROWN bound via sound GPU-resident resnet backward \
                     (#w4-gpu-dag-backward)"
                );
                match output_shape {
                    Some(ref shape) if shape.as_slice() != bounds.shape() => {
                        bounds.reshape(shape).unwrap_or_else(|_| bounds.clone())
                    }
                    _ => bounds,
                }
            }
            None => {
                // #dedup-root-collections Fix B: the CROWN backward pass used
                // to re-run its internal Step-1 intermediate collection over
                // the BIT-IDENTICAL root box that init just collected
                // (measured ~73 s of duplicate work per root episode on
                // vggnet16_2022 spec1; at tight budgets the recollection's
                // deadline gate then discarded everything for vacuous IBP).
                // Reuse the init map, but ONLY when it came from the same
                // collector Step 1 would run (or a strictly tighter one), so
                // no graph family gets weaker initial bounds than before:
                //   - CrownIbp: same collector as Step 1's per-node CROWN-IBP
                //     (or tighter than its >threshold IBP fallback). Only a
                //     conv DAG would route Step 1 to forward-linear instead —
                //     incomparable, so keep legacy behavior there.
                //   - ForwardLinear: may come from either a conv DAG or the
                //     dark sequential ConvTranspose reference lane. Reuse it
                //     only when Step 1 independently selects forward-linear;
                //     otherwise its collector is incomparable.
                //   - Ibp: Step 1 might upgrade to per-node CROWN-IBP (and
                //     its plain-IBP arm is the scalar f64 path, #4219), so
                //     preserve the legacy internal collection byte-for-byte.
                let step1_would_use_forward_linear = self.has_conv_layers()
                    && !self.is_sequential_graph(&exec_order)
                    && GraphNetwork::forward_linear_reference_enabled();
                let reuse_init_bounds = can_reuse_initial_node_bounds(
                    node_bounds_source,
                    step1_would_use_forward_linear,
                );
                self.propagate_crown_with_engine_and_deadline_and_node_bounds(
                    input,
                    engine,
                    config.deadline,
                    reuse_init_bounds.then(|| reference_bounds.current()),
                )?
                .bounds
            }
        };
        let mut best_lower: ArrayD<f32> = crown_bounds.lower().clone();
        let mut best_upper: ArrayD<f32> = crown_bounds.upper().clone();
        // Use finite-only sum to prevent -Inf poisoning the early-stopping metric (#2857).
        // Prior layout-agnostic fix: #1939.
        let mut best_lower_sum: f32 = finite_lower_sum(crown_bounds.lower());
        let mut prev_best_lower_sum = best_lower_sum;
        let mut no_improve_iters = 0usize;
        let mut lr = config.learning_rate;
        let mut infeasible_bounds: Option<BoundedTensor> = None;
        let mut total_gradient_skips: usize = 0;

        // The `Analytic` method takes its per-ReLU gradients directly from the CPU
        // backward's in-place fill; replacing that backward with the GPU warmup
        // bound would leave them zero (sound, but alpha would never move). Only
        // methods with their own gradient source (AnalyticChain via the GPU
        // warmup-gradient hook / SPSA / FiniteDifferences via bound evals) may take
        // the in-loop GPU bound. The PRE-LOOP initial bound above fills no
        // gradients, so it is exempt from this guard.
        // The GPU-resident warmup bound bypasses the CPU INVPROP seed augment, so
        // when the assume-violation gamma ascent is active we must use the CPU
        // backward (which applies the augment) for consistent bounds + gradient.
        // Gated on optimize_gammas (default off) => normal runs are unaffected.
        let in_loop_gpu_bound_ok = !(matches!(
            config.gradient_method,
            crate::bounds::GradientMethod::Analytic
        ) || (invprop_enabled && config.invprop.optimize_gammas));
        // A terminal optimizer update is dead only when the caller discards
        // the state. Root collection returns it for BaB warm-starting and
        // immediately re-evaluates it, so that path must retain the legacy
        // final gradient/update even when the experiment gate is enabled.
        let final_bound_only = result_use.terminal_bound_only(final_bound_only_gate);
        if final_bound_only_gate && !final_bound_only {
            debug!(
                ?result_use,
                "DAG α-CROWN: NY_ALPHA_FINAL_BOUND_ONLY fail-closed because returned state is observable"
            );
        }

        // #phase-telemetry (dark, NY_PHASE_TELEMETRY=1, print-only): phase
        // markers for the dag-alpha warmup loop — lever pricing needs phase
        // boundaries, not single-row wall deltas (~±15% layout noise across
        // builds). The gate is checked BEFORE every `format!` so the
        // default-unset path stays allocation-free and byte-identical.
        // `phase_iters_started` tracks how many iterations have BEGUN so every
        // exit path below can report an iterations-completed count.
        if crate::phase_telemetry::phase_telemetry_enabled() {
            crate::phase_telemetry::phase_marker(&format!(
                "dag-alpha-warmup loop-enter planned-iters={}",
                config.iterations
            ));
        }
        let mut phase_iters_started = 0usize;
        // #wall-refresh-cumulative: the configured fraction is one TOTAL
        // alpha-loop refresh pool, not a fresh geometric slice on every
        // improving iteration. The old `0.25 * remaining` recurrence consumed
        // 1-(0.75^N) of the root window (76% after five refreshes), starving
        // BaB. Every completed candidate remains a certified enclosure, and
        // exhaustion simply keeps the previous sound reference map. A run
        // without a global deadline preserves the historical unbounded
        // collection path.
        let alpha_refresh_fraction = alpha_refresh_fraction_from_env();
        let mut alpha_refresh_budget_remaining: Option<std::time::Duration> = None;
        for iter in 0..config.iterations {
            phase_iters_started = iter + 1;
            if crate::phase_telemetry::phase_telemetry_enabled() {
                crate::phase_telemetry::phase_marker(&format!(
                    "dag-alpha-warmup iter={iter} start"
                ));
            }
            // Deadline check (#2962): bail early if verification timeout budget
            // is exhausted. Return current best bounds instead of running all iterations.
            // Matches pattern in alpha_crown_loop.rs:112 and bounds/alpha.rs:114.
            if config.past_deadline() {
                info!(
                    "DAG α-CROWN: deadline exceeded at iteration {}/{}, returning best bounds",
                    iter, config.iterations
                );
                break;
            }
            let is_last_iter = iter == config.iterations - 1;
            let need_grad =
                alpha_iteration_needs_gradient(iter, config.iterations, final_bound_only);
            let node_bounds = reference_bounds.current();

            // Initialize gradients for each ReLU node
            let mut gradients: Vec<Array1<f32>> = if need_grad {
                relu_nodes
                    .iter()
                    .map(|(name, _)| {
                        let pre_act = self.relu_preactivation_bounds(
                            name,
                            input,
                            node_bounds,
                            "dag-alpha-gradient-init",
                        )?;
                        Ok(Array1::zeros(pre_act.len()))
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            // Separate upper-path gradient buffer (#3393).
            let mut gradients_upper: Vec<Array1<f32>> =
                gradients.iter().map(|g| Array1::zeros(g.len())).collect();

            // Run backward pass through DAG with alpha values.
            // Pass bilinear/mul_binary alphas so nonlinear nodes use interpolated bounds (#3287, #3439).
            let (bilinear_ref, mul_binary_ref) =
                gradients::alpha_refs(&ctx, &bilinear_alphas, &mul_binary_alphas);
            // #root-alpha-gpu (B): this iteration's loop-top GPU fold cache
            // (bound + local-rule gradients + segments from ONE kernel call).
            // Declared per-iteration so a previous iteration's fold can never
            // leak; consumed by the gradient site below. The reuse is valid
            // because α only changes at the END of the iteration
            // (`update_all_alphas` below), so the loop-top fold's α is still
            // current at the gradient site.
            let mut warmup_iter_cache: Option<gradients::WarmupGpuIterCache> = None;

            // #unsat-keystone: GPU-resident warmup BOUND fast-path (the #1 wall — the CPU
            // dag_alpha_backward_pass is ~7s/iter on cifar100 → warmup eats the whole
            // budget → 0 BaB domains). When it fires, gradients stay zero-initialized here
            // and are filled by the GPU warmup-gradient path at the gradient site below
            // (for a fully-decomposed resnet suffix that covers every ReLU). Gated, sound,
            // CPU fallback. Non-soundness-critical (warmup alpha).
            let mut concrete_bounds = match in_loop_gpu_bound_ok
                .then(|| {
                    if !need_grad {
                        self.try_gpu_warmup_bound_only(&ctx, node_bounds, &runtime)
                    } else if root_alpha_gpu_enabled() {
                        // #root-alpha-gpu (B): take the FULL fold once and keep
                        // its cache for the gradient site (one GPU fold per
                        // iteration instead of two). Bound value identical to
                        // the wrapper below by construction.
                        self.try_gpu_warmup_bound_full(&ctx, node_bounds, &runtime)
                            .map(|(bounds, mut cache)| {
                                cache.iter = iter;
                                warmup_iter_cache = Some(cache);
                                bounds
                            })
                    } else {
                        self.try_gpu_warmup_bound(&ctx, node_bounds, &runtime)
                    }
                })
                .flatten()
            {
                Some(bounds) => bounds,
                None => {
                    if need_grad {
                        self.dag_alpha_backward_pass_with_engine(
                            input,
                            node_bounds,
                            &exec_order,
                            output_dim,
                            input_dim,
                            runtime.relu_name_to_idx(),
                            runtime.graph(),
                            runtime.invprop(),
                            &mut gradients,
                            &mut gradients_upper,
                            engine,
                            bilinear_ref,
                            mul_binary_ref,
                            config.deadline,
                        )?
                    } else {
                        self.dag_alpha_bound_pass_with_engine(
                            input,
                            node_bounds,
                            &exec_order,
                            output_dim,
                            input_dim,
                            runtime.relu_name_to_idx(),
                            runtime.graph(),
                            runtime.invprop(),
                            engine,
                            bilinear_ref,
                            mul_binary_ref,
                            config.deadline,
                        )?
                    }
                }
            };

            if let Some(state) = runtime.invprop_mut() {
                if bounds_infeasible(&concrete_bounds) {
                    state.mark_infeasible(0)?;
                    state.apply_infeasible_mask(&mut concrete_bounds);
                    infeasible_bounds = Some(concrete_bounds);
                    break;
                }
            }

            // Update element-wise best bounds with layout-agnostic iteration.
            // This handles non-standard layout arrays and shape-only ndim mismatch
            // as long as element counts match (#2076, #2087).
            // Skip during warmup window to avoid locking in noisy early-iteration bounds.
            // Matches α,β-CROWN's start_save_best (optimized_bounds.py:785-797).
            if config.should_save_best(iter, is_last_iter) {
                update_elementwise_best_bounds(
                    &mut best_lower,
                    &mut best_upper,
                    &concrete_bounds,
                    iter,
                )?;
            }

            // Finite-only sum for early stopping (#2857). Layout-agnostic (#1939).
            let lower_sum: f32 = finite_lower_sum(concrete_bounds.lower());

            // NaN detection: if any bound element is NaN, the backward pass produced
            // garbage. Break early to avoid wasting remaining iterations — the
            // post-loop has_nan check will fall back to CROWN. (#2597)
            if concrete_bounds.lower().iter().any(|v| v.is_nan())
                || concrete_bounds.upper().iter().any(|v| v.is_nan())
            {
                warn!(
                    "DAG α-CROWN: NaN in bounds at iteration {iter}, aborting optimization (#2597)"
                );
                break;
            }

            // #invprop-alpha-budget: mid-iteration deadline check. The loop-top
            // check cannot see a backward pass that overran the budget INSIDE
            // this iteration; without this the iteration would go on to spend
            // more past-deadline time on gradient probes and the reference-
            // bound refresh before the next loop-top check fires. Sound: only
            // stops optimizing sooner, returning the elementwise best bounds.
            if config.past_deadline() {
                if !config.should_save_best(iter, is_last_iter) {
                    update_elementwise_best_bounds(
                        &mut best_lower,
                        &mut best_upper,
                        &concrete_bounds,
                        iter,
                    )?;
                }
                info!(
                    "DAG α-CROWN: deadline exceeded after iteration {}/{} backward pass, \
                     returning best bounds",
                    iter, config.iterations
                );
                break;
            }

            // Spec-proven early-exit (#warmup-early-exit). When the single-objective
            // warmup carries a `spec_early_exit`, project the elementwise BEST output
            // bounds (the bounds this loop will actually return) onto the objective and
            // stop the moment they already prove the property against the threshold.
            // SOUND: this only stops optimizing sooner — the projected bound at the exit
            // iteration is a valid over-approximation that clears the threshold; no bound
            // *computation* changes, and `None` callers (every non-warmup caller) skip
            // this entirely.
            if let Some(spec) = config.spec_early_exit.as_ref() {
                if let (Some(lo), Some(hi)) = (best_lower.as_slice(), best_upper.as_slice()) {
                    if let Some((root_lower, root_upper)) = spec.project_bounds(lo, hi) {
                        if spec.is_verified(root_lower, root_upper) {
                            debug!(
                                "DAG α-CROWN: spec-proven early-exit at iter {} \
                                 (root bound [{:.4}, {:.4}] clears threshold {})",
                                iter, root_lower, root_upper, spec.threshold
                            );
                            break;
                        }
                    }
                }
            }

            // Track best lower_sum for early stopping
            let improved_output = lower_sum > best_lower_sum;
            if improved_output {
                best_lower_sum = lower_sum;
            }

            // INVPROP: projected gamma ascent on the output-seed duals (DAG path,
            // Stage 3). Gated on optimize_gammas (default OFF => normal runs untouched).
            // Reuses the flat all_ny_params/update_ny_params interface; under the
            // output-node-only default only the seed duals are active, so the SPSA probe
            // steers exactly those. Soundness is gamma-independent (the augment is sound
            // for any gamma>=0 and best-bounds keeps the tightest), so a cheap
            // deterministic one-sided SPSA estimate suffices; the probe backward is a
            // pure gradient estimate whose bounds are discarded.
            // THROUGHPUT GUARD (see alpha_crown_loop): each probe is one extra
            // backward — expensive on conv-resnets — so cap it to the first few
            // iters and skip near the deadline, so on-by-default can only help or
            // no-op, never turn a budget into a timeout.
            const INVPROP_ASCENT_MAX_ITERS: usize = 5;
            if invprop_enabled
                && config.invprop.optimize_gammas
                && need_grad
                && iter < INVPROP_ASCENT_MAX_ITERS
                && !config.past_deadline()
                && lower_sum.is_finite()
            {
                let base_params: Vec<f32> = runtime
                    .invprop()
                    .map(|s| s.all_ny_params())
                    .unwrap_or_default();
                if !base_params.is_empty() {
                    let lr_g = if config.invprop.gamma_lr > 0.0 {
                        config.invprop.gamma_lr
                    } else {
                        0.5
                    };
                    let delta = 0.1f32;
                    let sign = |i: usize| -> f32 {
                        if ((iter.wrapping_mul(2_654_435_761) ^ i.wrapping_mul(40_503)) & 1) == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    };
                    let perturbed: Vec<f32> = base_params
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (v + delta * sign(i)).max(0.0))
                        .collect();
                    let mut probe_ok = false;
                    if let Some(state) = runtime.invprop_mut() {
                        probe_ok = state.update_ny_params(&perturbed).is_ok();
                    }
                    if probe_ok {
                        let mut g0: Vec<Array1<f32>> =
                            gradients.iter().map(|g| Array1::zeros(g.len())).collect();
                        let mut g1: Vec<Array1<f32>> =
                            g0.iter().map(|g| Array1::zeros(g.len())).collect();
                        let (blr, mbr) =
                            gradients::alpha_refs(&ctx, &bilinear_alphas, &mul_binary_alphas);
                        let obj_plus = match self.dag_alpha_backward_pass_with_engine(
                            input,
                            node_bounds,
                            &exec_order,
                            output_dim,
                            input_dim,
                            runtime.relu_name_to_idx(),
                            runtime.graph(),
                            runtime.invprop(),
                            &mut g0,
                            &mut g1,
                            engine,
                            blr,
                            mbr,
                            config.deadline,
                        ) {
                            Ok(b) => finite_lower_sum(b.lower()),
                            Err(_) => lower_sum,
                        };
                        let scale = (obj_plus - lower_sum) / delta;
                        let updated: Vec<f32> = if scale.is_finite() {
                            base_params
                                .iter()
                                .enumerate()
                                .map(|(i, &v)| (v + lr_g * scale * sign(i)).max(0.0))
                                .collect()
                        } else {
                            base_params
                        };
                        if let Some(state) = runtime.invprop_mut() {
                            let _ = state.update_ny_params(&updated);
                        }
                    }
                }
            }

            // Early stopping check (compare best improvement since last iteration).
            let best_improvement = best_lower_sum - prev_best_lower_sum;
            if best_improvement < config.tolerance {
                no_improve_iters += 1;
            } else {
                no_improve_iters = 0;
            }
            if iter > 0 && no_improve_iters >= config.early_stop_patience {
                if !config.should_save_best(iter, false) {
                    update_elementwise_best_bounds(
                        &mut best_lower,
                        &mut best_upper,
                        &concrete_bounds,
                        iter,
                    )?;
                }
                info!(
                    "DAG α-CROWN: Converged at iteration {} (best improvement < {} for {} iters)",
                    iter, config.tolerance, no_improve_iters
                );
                break;
            }

            // Pilot iteration check: after SECOND iteration, verify α-CROWN helps.
            // Must be iter >= 1 (not iter == 0) because iteration 0 uses CROWN-initialized
            // alpha values and always produces bounds identical to plain CROWN. The first
            // alpha update happens at the END of iter 0, so iter 1 is the first iteration
            // that reflects optimized alpha values. (#3293)
            if iter == 1 && config.adaptive_skip && config.adaptive_skip_pilot {
                // Compute improvement over initial CROWN bounds (#2857, #1939).
                let initial_lower_sum: f32 = finite_lower_sum(crown_bounds.lower());
                let pilot_improvement = best_lower_sum - initial_lower_sum;

                if pilot_improvement < config.pilot_improvement_threshold {
                    info!(
                        "DAG α-CROWN: Pilot iteration improvement ({:.3e}) < threshold ({:.3e}). \
                         α-CROWN optimization is not helping, skipping remaining iterations.",
                        pilot_improvement, config.pilot_improvement_threshold
                    );
                    // Return best bounds found so far (CROWN or pilot iteration bounds)
                    let mut pilot_lower = best_lower.clone();
                    let mut pilot_upper = best_upper.clone();
                    let widened = clamp_inverted_best_bounds(
                        &mut pilot_lower,
                        &mut pilot_upper,
                        "dag-alpha-crown-pilot-exit",
                    );
                    if widened > 0 {
                        // Fall back to CROWN bounds for inverted elements (#3754).
                        for (best, &crown) in
                            pilot_lower.iter_mut().zip(crown_bounds.lower().iter())
                        {
                            if !best.is_finite() {
                                *best = crown;
                            }
                        }
                        for (best, &crown) in
                            pilot_upper.iter_mut().zip(crown_bounds.upper().iter())
                        {
                            if !best.is_finite() {
                                *best = crown;
                            }
                        }
                    }
                    let bounds = BoundedTensor::new(pilot_lower, pilot_upper).map_err(|e| {
                        NyError::InternalError(format!(
                            "DAG α-CROWN pilot bounds invalid after CROWN fallback: {e}"
                        ))
                    })?;
                    // #phase-telemetry: this pilot skip returns without
                    // reaching the shared post-loop exit marker below.
                    if crate::phase_telemetry::phase_telemetry_enabled() {
                        crate::phase_telemetry::phase_marker(&format!(
                            "dag-alpha-warmup loop-exit iters={phase_iters_started} (pilot-skip)"
                        ));
                    }
                    return Ok((bounds, runtime.snapshot_graph()));
                } else {
                    debug!(
                        "DAG α-CROWN: Pilot iteration improvement ({:.3e}) >= threshold ({:.3e}). \
                         Continuing optimization.",
                        pilot_improvement, config.pilot_improvement_threshold
                    );
                }
            }

            // All terminal bound validity, best-bound, early-stop, and pilot
            // bookkeeping is complete. Preserve the exact alpha/reference/
            // optimizer state that produced it; nothing below can feed another
            // evaluated bound.
            if !need_grad {
                debug!(
                    method = ?config.gradient_method,
                    iter,
                    skipped_gradient_dispatches = 1usize,
                    skipped_state_updates = 1usize,
                    "DAG α-CROWN: NY_ALPHA_FINAL_BOUND_ONLY terminal pass"
                );
                break;
            }

            let refresh_candidate =
                if iter >= 1 && improved_output && !reference_bounds.targets().is_empty() {
                    // Carry forward tighter activation-input bounds between
                    // iterations, matching alpha-beta-CROWN's
                    // `best_intermediate_bounds` / `reference_bounds`
                    // tightening for optimizable activations.
                    // Source: auto_LiRPA `optimized_bounds.py:338-367,500-615`.
                    //
                    // #w4-gpu-dag-backward: bound the refresh to a SHARE of the
                    // remaining budget instead of the full global deadline.
                    // Measured (cifar100 resnet-medium, release): one unbounded
                    // refresh ran ~120s of per-target spec-batched CROWN requests
                    // — past the whole 95s timeout — while a GPU warmup iteration
                    // costs ~1.5s, so a single refresh starved every remaining
                    // iteration AND BaB. On expiry the refresh falls back to the
                    // previous (sound) reference bounds for outstanding targets,
                    // so capping only trades tightness for schedule — never
                    // soundness.
                    // #wall-airtime: all improving iterations draw from ONE
                    // cumulative pool. `NY_ALPHA_REFRESH_FRACTION` controls the
                    // pool's share; unset uses the shipped 0.25 default.
                    let refresh_start = std::time::Instant::now();
                    let refresh_deadline = config.deadline.map(|global_deadline| {
                        let global_remaining = global_deadline
                            .checked_duration_since(refresh_start)
                            .unwrap_or_default();
                        let allowance = cumulative_alpha_refresh_allowance(
                            &mut alpha_refresh_budget_remaining,
                            global_remaining,
                            alpha_refresh_fraction,
                        );
                        refresh_start + allowance
                    });
                    let has_refresh_budget = config.deadline.is_none()
                        || refresh_deadline.is_some_and(|deadline| deadline > refresh_start);
                    if has_refresh_budget {
                        let candidate = self.collect_selected_crown_bounds_with_alpha(
                            input,
                            reference_bounds.targets(),
                            node_bounds,
                            runtime.graph(),
                            engine,
                            refresh_deadline,
                        )?;
                        debit_alpha_refresh_budget(
                            &mut alpha_refresh_budget_remaining,
                            refresh_start.elapsed(),
                        );
                        Some(candidate)
                    } else {
                        debug!(
                            iter,
                            "DAG α-CROWN: cumulative reference-refresh budget exhausted"
                        );
                        None
                    }
                } else {
                    None
                };

            // #root-alpha-gpu (B): a reference-bound refresh ran this
            // iteration — invalidate the loop-top fold's gradient reuse so the
            // gradient site re-folds fresh (a hygiene choice, not soundness:
            // gradients only steer α and never decide a verdict).
            if refresh_candidate.is_some() {
                if let Some(cache) = warmup_iter_cache.as_mut() {
                    cache.refresh_fired = true;
                }
            }

            // Compute gradients using configured method (SPSA, FD, Analytic, AnalyticChain).
            let eps = 1e-3;
            let mut grad_result = self.compute_dag_gradients(
                &ctx,
                node_bounds,
                &mut runtime,
                &mut bilinear_alphas,
                &mut mul_binary_alphas,
                &gradients,
                &gradients_upper,
                eps,
                iter,
                warmup_iter_cache.as_ref(),
            )?;
            self.compute_spsa_supplements(
                input,
                node_bounds,
                &exec_order,
                output_dim,
                input_dim,
                config,
                engine,
                &mut runtime,
                &gradients,
                &bilinear_alphas,
                &mut mul_binary_alphas,
                &mut grad_result.mul_binary_grads,
                &mut grad_result.s_shaped_grads,
                &mut grad_result.sqrt_grads,
                &mut grad_result.reciprocal_grads,
                has_bilinear,
                has_mul_binary,
                has_s_shaped,
                has_sqrt,
                has_reciprocal,
                eps,
                iter,
            )?;

            // Destructure to separate immutable and mutable borrows (#2297).
            // `numerical_gradients_upper` is `None` for non-Analytic methods,
            // avoiding a full Vec<Array1<f32>> clone per iteration.
            let gradients::GradientDispatchResult {
                numerical_gradients: ref lower_grads,
                numerical_gradients_upper: ref upper_grads_opt,
                ref s_shaped_grads,
                ref sqrt_grads,
                ref reciprocal_grads,
                bilinear_grads: ref mut bl_grads,
                mul_binary_grads: ref mut mb_grads,
            } = grad_result;
            let upper_grads: &[Array1<f32>] = upper_grads_opt.as_deref().unwrap_or(lower_grads);
            alpha_update::update_all_alphas(
                &mut runtime,
                config,
                lower_grads,
                upper_grads,
                s_shaped_grads,
                sqrt_grads,
                reciprocal_grads,
                &mut bilinear_alphas,
                bl_grads,
                &mut bilinear_adam_m,
                &mut bilinear_adam_v,
                &mut mul_binary_alphas,
                mb_grads,
                &mut mul_binary_adam_m,
                &mut mul_binary_adam_v,
                has_bilinear,
                has_mul_binary,
                invprop_enabled,
                lr,
                iter,
                &mut total_gradient_skips,
            )?;

            if let Some(candidate) = refresh_candidate {
                let tightened_targets = reference_bounds.merge_candidate(&candidate)?;
                reference_bounds.promote_best_to_current()?;
                debug!(
                    "DAG α-CROWN iter {}: refreshed {} activation-input reference targets",
                    iter, tightened_targets
                );
            }

            // Learning rate decay
            lr *= config.lr_decay;

            if iter % 5 == 0 {
                diagnostics::log_iteration_telemetry(
                    &runtime,
                    iter,
                    best_lower_sum,
                    prev_best_lower_sum,
                    lower_sum,
                    lr,
                );
            }

            // #invprop-alpha-budget: info-level progress heartbeat. The alpha
            // phase used to emit NOTHING at default log level between the
            // "Starting optimization" line and its exit — on slow models it
            // was indistinguishable from a hang. Every 10 iterations, report
            // position and remaining budget.
            if iter % 10 == 0 {
                match config
                    .deadline
                    .map(|d| d.saturating_duration_since(std::time::Instant::now()))
                {
                    Some(remaining) => info!(
                        "DAG α-CROWN: iter {}/{}: lower_sum={:.6}, best_lower_sum={:.6}, \
                         budget remaining {:.1}s",
                        iter,
                        config.iterations,
                        lower_sum,
                        best_lower_sum,
                        remaining.as_secs_f32()
                    ),
                    None => info!(
                        "DAG α-CROWN: iter {}/{}: lower_sum={:.6}, best_lower_sum={:.6}",
                        iter, config.iterations, lower_sum, best_lower_sum
                    ),
                }
            }

            prev_best_lower_sum = best_lower_sum;
        }

        // #phase-telemetry: shared loop exit (normal completion and every
        // `break` — deadline, converged, NaN, spec-early-exit, infeasible).
        // `?` error exits skip it, but those abort the pipeline anyway.
        if crate::phase_telemetry::phase_telemetry_enabled() {
            crate::phase_telemetry::phase_marker(&format!(
                "dag-alpha-warmup loop-exit iters={phase_iters_started}"
            ));
        }

        diagnostics::log_gradient_skip_summary(
            total_gradient_skips,
            config.iterations,
            runtime.relu_nodes().len(),
        );

        if let Some(bounds) = infeasible_bounds {
            return Ok((bounds, runtime.into_graph_alpha_state()));
        }

        // Return element-wise best bounds found across all iterations.
        // Fall back to CROWN only when NaN is present (computation error).
        // Infinite bounds are sound overapproximations from inversion widening
        // (clamp_inverted_best_bounds sets inverted elements to [-inf, +inf]).
        // The previous is_finite() check incorrectly discarded ALL optimization
        // progress when any single element was infinite (#2854).
        let alpha_state = runtime.into_graph_alpha_state();
        let has_nan =
            best_lower.iter().any(|v| v.is_nan()) || best_upper.iter().any(|v| v.is_nan());

        if !has_nan {
            // Clamp any inverted intervals from cross-iteration elementwise merge.
            let widened =
                clamp_inverted_best_bounds(&mut best_lower, &mut best_upper, "dag-alpha-crown");

            if widened > 0 {
                // Fall back to CROWN bounds for inverted elements (#3754).
                // Cross-iteration elementwise merge can produce inversions on DAG
                // topologies (e.g. diamond: branches with zero weights cause SPSA
                // noise to oscillate alpha slopes). The initial CROWN bounds are
                // sound and finite for finite-weight networks, so restoring them
                // for widened elements preserves soundness while avoiding -inf.
                for (best, &crown) in best_lower.iter_mut().zip(crown_bounds.lower().iter()) {
                    if !best.is_finite() {
                        *best = crown;
                    }
                }
                for (best, &crown) in best_upper.iter_mut().zip(crown_bounds.upper().iter()) {
                    if !best.is_finite() {
                        *best = crown;
                    }
                }
            }

            let bounds = BoundedTensor::new(best_lower, best_upper).map_err(|e| {
                NyError::InternalError(format!(
                    "DAG α-CROWN best bounds invalid after CROWN fallback: {e}"
                ))
            })?;
            Ok((bounds, alpha_state))
        } else {
            // Fall back to CROWN if NaN detected (actual computation error)
            warn!("DAG α-CROWN: NaN in best bounds, falling back to plain CROWN");
            let bounds = self
                .propagate_crown_with_engine_and_deadline(input, engine, config.deadline)
                .map(|r| r.bounds)?;
            Ok((bounds, alpha_state))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        can_reuse_initial_node_bounds, cumulative_alpha_refresh_allowance,
        debit_alpha_refresh_budget, parse_alpha_refresh_fraction, AlphaReferenceBoundsSource,
        DagAlphaLoopResultUse, DEFAULT_ALPHA_REFRESH_FRACTION, MIN_ALPHA_REFRESH_FRACTION,
    };
    use std::time::Duration;

    #[test]
    fn terminal_bound_only_requires_dead_returned_state() {
        assert!(!DagAlphaLoopResultUse::BoundsOnly.terminal_bound_only(false));
        assert!(DagAlphaLoopResultUse::BoundsOnly.terminal_bound_only(true));
        assert!(!DagAlphaLoopResultUse::BoundsAndState.terminal_bound_only(false));
        assert!(!DagAlphaLoopResultUse::BoundsAndState.terminal_bound_only(true));
    }

    #[test]
    fn dedup_reuse_requires_the_same_step1_collector() {
        assert!(can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::CrownIbp,
            false
        ));
        assert!(!can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::CrownIbp,
            true
        ));
        assert!(can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::ForwardLinear,
            true
        ));
        assert!(!can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::ForwardLinear,
            false
        ));
        assert!(!can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::Ibp,
            false
        ));
        assert!(!can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::Ibp,
            true
        ));
    }

    #[test]
    fn alpha_refresh_fraction_defaults_for_absent_or_invalid_values() {
        assert_eq!(
            parse_alpha_refresh_fraction(None),
            DEFAULT_ALPHA_REFRESH_FRACTION
        );
        for raw in [
            "",
            "not-a-number",
            "NaN",
            "inf",
            "-inf",
            "0",
            "-0.5",
            "0.009",
            "1.001",
        ] {
            assert_eq!(
                parse_alpha_refresh_fraction(Some(raw)),
                DEFAULT_ALPHA_REFRESH_FRACTION,
                "raw={raw:?}"
            );
        }
    }

    #[test]
    fn alpha_refresh_fraction_accepts_in_range_values_and_boundaries() {
        for (raw, expected) in [
            ("0.01", MIN_ALPHA_REFRESH_FRACTION),
            (" 0.125 ", 0.125),
            ("0.25", DEFAULT_ALPHA_REFRESH_FRACTION),
            ("1", 1.0),
        ] {
            assert_eq!(
                parse_alpha_refresh_fraction(Some(raw)),
                expected,
                "raw={raw:?}"
            );
        }
    }

    #[test]
    fn alpha_refresh_budget_is_cumulative_and_saturates_on_exhaustion() {
        let mut budget = None;
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut budget,
                Duration::from_secs(100),
                DEFAULT_ALPHA_REFRESH_FRACTION,
            ),
            Duration::from_secs(25)
        );

        debit_alpha_refresh_budget(&mut budget, Duration::from_secs(10));
        // A fresh 25%-of-remaining envelope would grant 22.5s here. The
        // cumulative policy exposes only the 15s left in the original pool.
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut budget,
                Duration::from_secs(90),
                DEFAULT_ALPHA_REFRESH_FRACTION,
            ),
            Duration::from_secs(15)
        );

        // The global verifier deadline remains an independent hard ceiling.
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut budget,
                Duration::from_secs(4),
                DEFAULT_ALPHA_REFRESH_FRACTION,
            ),
            Duration::from_secs(4)
        );

        debit_alpha_refresh_budget(&mut budget, Duration::from_secs(20));
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut budget,
                Duration::from_secs(70),
                DEFAULT_ALPHA_REFRESH_FRACTION,
            ),
            Duration::ZERO,
            "collector overrun must exhaust rather than replenish the pool"
        );
    }

    #[test]
    fn exhausted_refresh_budget_does_not_enable_state_discard_shortcuts() {
        let mut budget = None;
        let allowance = cumulative_alpha_refresh_allowance(
            &mut budget,
            Duration::from_secs(8),
            DEFAULT_ALPHA_REFRESH_FRACTION,
        );
        debit_alpha_refresh_budget(&mut budget, allowance);
        assert_eq!(
            cumulative_alpha_refresh_allowance(
                &mut budget,
                Duration::from_secs(6),
                DEFAULT_ALPHA_REFRESH_FRACTION,
            ),
            Duration::ZERO
        );

        // Exhausting a schedule-only reference-refresh pool must not make the
        // returned optimizer state dead. Collection still retains its terminal
        // update for immediate BaB re-evaluation.
        assert!(
            !DagAlphaLoopResultUse::BoundsAndState.terminal_bound_only(true),
            "state-returning DAG collection must continue to fail closed"
        );
    }
}
