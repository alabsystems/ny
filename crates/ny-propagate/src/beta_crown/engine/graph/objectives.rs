// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::trace;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::domain::GraphCrownContext;
use crate::beta_crown::domain::MultiObjectiveTargets;
use crate::beta_crown::state::GraphBetaState;
use crate::bounds::{nan_propagating_max, nan_propagating_min};
use std::sync::Arc;

use crate::GraphNetwork;

use super::super::BetaCrownVerifier;
#[cfg(test)]
use super::MultiObjectiveResult;

type MultiObjectiveWarmStartResult = (
    Vec<(f32, f32)>,
    std::collections::HashMap<String, Arc<BoundedTensor>>,
    Vec<Option<CachedLinearBounds>>,
);

/// Compute scalar objective bounds from output tensor using interval arithmetic.
///
/// Given an output `BoundedTensor` and a linear objective vector `c`, computes
/// `[lower, upper]` bounds on `c^T y` where `y` ranges over the output interval.
///
/// Uses the standard interval arithmetic rule: for coefficient `c_i >= 0`, accumulate
/// `c_i * lower_i` for the lower bound and `c_i * upper_i` for the upper bound;
/// for `c_i < 0`, swap lower and upper.
///
/// SOUNDNESS (#concretize-soundness-hardening): this result feeds
/// `domain_is_verified` at the root early-exit, where an inward round-to-nearest
/// endpoint would be an undetectable, terminal false Verified. Accumulate in f64
/// — `f32 x f32` promoted to f64 is exact (48 < 53 significand bits), so only
/// the additions round — then close with a directed f32 cast, mirroring
/// `LinearBounds::concretize_sound`.
///
/// ENSURES: `lower <= c^T y <= upper` for every `y` in the output box.
pub(super) fn objective_bounds(output: &BoundedTensor, objective: &[f32]) -> Result<(f32, f32)> {
    let flat = output.flatten();
    if flat.len() != objective.len() {
        return Err(NyError::shape_mismatch(
            vec![objective.len()],
            vec![flat.len()],
        ));
    }

    let mut lower = 0.0f64;
    let mut upper = 0.0f64;
    for (idx, &c) in objective.iter().enumerate() {
        let c = c as f64;
        let l = flat.lower()[[idx]] as f64;
        let u = flat.upper()[[idx]] as f64;
        if c >= 0.0 {
            lower += c * l;
            upper += c * u;
        } else {
            lower += c * u;
            upper += c * l;
        }
    }
    // Guard against NaN from degenerate CROWN propagation (#2359).
    // NaN comparisons always return false in Rust, so NaN bounds would create
    // zombie domains that re-enter the BaB queue indefinitely without converging.
    // Conservative fallback: treat as unbounded so BaB can split or discard.
    if lower.is_nan() || upper.is_nan() {
        return Ok((f32::NEG_INFINITY, f32::INFINITY));
    }
    Ok((next_down_f32(lower as f32), next_up_f32(upper as f32)))
}

fn build_spec_matrix(objectives: &[Vec<f32>]) -> Option<ndarray::Array2<f32>> {
    if objectives.is_empty() {
        return None;
    }
    let num_specs = objectives.len();
    let output_dim = objectives[0].len();
    let mut data = Vec::with_capacity(num_specs * output_dim);
    for objective in objectives {
        if objective.len() != output_dim {
            return None;
        }
        data.extend_from_slice(objective);
    }
    ndarray::Array2::from_shape_vec((num_specs, output_dim), data).ok()
}

fn spec_bounds_to_vec(bounds: &BoundedTensor) -> Vec<(f32, f32)> {
    let flat = bounds.flatten();
    (0..flat.len())
        .map(|idx| (flat.lower()[[idx]], flat.upper()[[idx]]))
        .collect()
}

fn split_captured_multi_row_cache(
    captured_cache: Option<CachedLinearBounds>,
    num_objectives: usize,
) -> Vec<Option<CachedLinearBounds>> {
    captured_cache
        .and_then(|cache| cache.split_multi_row(num_objectives))
        .map(|per_objective| per_objective.into_iter().map(Some).collect())
        .unwrap_or_else(|| vec![None; num_objectives])
}

impl BetaCrownVerifier {
    /// Compute bounds for multiple objectives from output tensor using interval arithmetic.
    ///
    /// **Note**: This uses post-hoc interval arithmetic which loses output correlations.
    /// For tighter bounds, use spec-guided CROWN via `propagate_multi_objective_spec_guided`.
    pub(super) fn objective_bounds_multi(
        output: &BoundedTensor,
        objectives: &[Vec<f32>],
    ) -> Result<Vec<(f32, f32)>> {
        objectives
            .iter()
            .map(|obj| objective_bounds(output, obj))
            .collect()
    }

    /// Propagate bounds with β for multi-objective verification without optimization.
    ///
    /// This is used for deep domains where we skip β optimization and rely on
    /// inherited β values from warmup.
    ///
    /// Uses **spec-guided CROWN** with a dense multi-row spec matrix to preserve
    /// output correlations across all objectives in one backward pass, resulting
    /// in tighter bounds compared to post-hoc interval arithmetic.
    /// See issue #593 and docs/ANALYSIS-resnet-verification-gap-root-cause-2026-01-07.md.
    #[cfg(test)]
    pub(super) fn propagate_multi_objective_with_beta(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
    ) -> Result<MultiObjectiveResult> {
        let seed_caches = vec![None; targets.objectives.len()];
        let (obj_bounds, final_node_bounds, _cached_las) = self
            .propagate_multi_objective_with_beta_and_cache(
                graph,
                input,
                context,
                beta_state,
                targets,
                &seed_caches,
                false,
            )?;
        Ok((obj_bounds, final_node_bounds))
    }

    // Justification: this helper threads graph context, beta state, objective
    // set, optional warm-start caches, and the capture flag together.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn propagate_multi_objective_with_beta_and_cache(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
        seed_caches: &[Option<&CachedLinearBounds>],
        capture_caches: bool,
    ) -> Result<MultiObjectiveWarmStartResult> {
        if seed_caches.len() != targets.objectives.len() {
            return Err(NyError::InvalidSpec(format!(
                "multi-objective seed cache length {} != objective length {} (#3813)",
                seed_caches.len(),
                targets.objectives.len()
            )));
        }

        let spec_matrix = build_spec_matrix(targets.objectives).ok_or_else(|| {
            NyError::InvalidSpec(
                "multi-objective dense spec matrix must be non-empty and rectangular".to_string(),
            )
        })?;
        let combined_seed_cache = seed_caches
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .and_then(|caches| CachedLinearBounds::stack_single_row(&caches));
        let (output, node_bounds, captured_cache) = self
            .propagate_crown_with_graph_beta_and_spec_matrix(
                graph,
                input,
                context,
                beta_state,
                &spec_matrix,
                combined_seed_cache.as_ref(),
                capture_caches,
            )?;

        Ok((
            spec_bounds_to_vec(&output),
            node_bounds,
            split_captured_multi_row_cache(captured_cache, targets.objectives.len()),
        ))
    }

    /// Optimize β parameters using analytical gradients for multi-objective verification.
    ///
    /// Computes gradients analytically from the A matrices, avoiding the 3 forward
    /// passes per iteration that SPSA requires (~3x faster).
    ///
    /// For each iteration:
    /// 1. Propagate bounds and capture A matrices (1 forward pass)
    /// 2. Compute objective bounds for all objectives
    /// 3. Find the critical objective (min or max margin among unverified)
    /// 4. Compute β gradients for the critical objective using A matrices
    /// 5. Adam gradient step
    ///
    /// When `conjunctive` is true, optimizes the **maximum** margin instead of minimum.
    /// For conjunctive (AND) properties, only ONE objective needs to exceed its threshold
    /// to verify the domain, so we optimize the best objective rather than the worst.
    /// Source: designs/2026-03-05-joint-conjunctive-bab.md, #3334.
    #[cfg(test)]
    pub(super) fn optimize_graph_beta_analytical_multi_objective(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
        conjunctive: bool,
    ) -> Result<MultiObjectiveResult> {
        let seed_caches = vec![None; targets.objectives.len()];
        let (obj_bounds, node_bounds, _cached_las) = self
            .optimize_graph_beta_analytical_multi_objective_with_cache(
                graph,
                input,
                context,
                beta_state,
                targets,
                conjunctive,
                &seed_caches,
                false,
            )?;
        Ok((obj_bounds, node_bounds))
    }

    // Justification: analytical beta optimization needs graph/input/context,
    // mutable beta state, multi-objective targets, conjunctive mode, and cache controls.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn optimize_graph_beta_analytical_multi_objective_with_cache(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
        conjunctive: bool,
        seed_caches: &[Option<&CachedLinearBounds>],
        capture_caches: bool,
    ) -> Result<MultiObjectiveWarmStartResult> {
        // Skip if no beta parameters or iterations disabled
        if beta_state.is_empty() || self.config.beta_iterations == 0 {
            // Use spec-guided CROWN for tighter bounds (preserves output correlations)
            return self.propagate_multi_objective_with_beta_and_cache(
                graph,
                input,
                context,
                beta_state,
                targets,
                seed_caches,
                capture_caches,
            );
        }

        let spec_matrix = build_spec_matrix(targets.objectives).ok_or_else(|| {
            NyError::InvalidSpec(
                "multi-objective dense spec matrix must be non-empty and rectangular".to_string(),
            )
        })?;

        // Compute margin metric for optimization.
        // Disjunctive: min margin (all objectives must be verified → optimize worst).
        // Conjunctive: max margin (any objective verified suffices → optimize best). #3334
        // Defense-in-depth: assert lengths match before triple .zip() (#3383).
        debug_assert_eq!(
            targets.objectives.len(),
            targets.thresholds.len(),
            "compute_margin: objectives/thresholds length mismatch (#3383)"
        );
        debug_assert_eq!(
            targets.objectives.len(),
            targets.verified_mask.len(),
            "compute_margin: objectives/verified_mask length mismatch (#3383)"
        );
        let compute_margin = |bounds: &[(f32, f32)]| -> f32 {
            debug_assert_eq!(
                bounds.len(),
                targets.thresholds.len(),
                "compute_margin: bounds/thresholds length mismatch (#3383)"
            );
            let margins = bounds
                .iter()
                .zip(targets.thresholds.iter())
                .zip(targets.verified_mask.iter())
                .filter(|((_, _), &v)| !v) // Only unverified objectives
                .map(|(((l, _), &t), _)| l - t);
            if conjunctive {
                margins.fold(f32::NEG_INFINITY, nan_propagating_max)
            } else {
                margins.fold(f32::INFINITY, nan_propagating_min) // #2577: NaN margin must propagate
            }
        };

        let mut best_margin = f32::NEG_INFINITY;
        let mut best_beta_snapshot: Option<GraphBetaState> = None;

        // Best fully-computed loop result captured alongside the best β snapshot.
        // On a mid-loop deadline we return THIS directly instead of running the
        // (expensive) final spec-guided pass + snapshot evaluation below. These
        // bounds come from `propagate_crown_with_graph_beta_and_spec_matrix_*`
        // over the same dense spec matrix as the final pass, so they are valid
        // sound spec-guided CROWN bounds — we only stop optimizing sooner. (#3109)
        type LoopBest = (
            Vec<(f32, f32)>,
            std::collections::HashMap<String, Arc<BoundedTensor>>,
        );
        let mut best_loop_result: Option<LoopBest> = None;

        // Periodic β snapshots for spec-guided evaluation at the end.
        // The post-hoc margin used during the loop is only a proxy and can be
        // β-insensitive for certain architectures (e.g., when objectives use
        // negative output coefficients whose lower bound depends on the output
        // UPPER relaxation, which β doesn't improve). Saving periodic snapshots
        // ensures we evaluate the optimal β that post-hoc tracking may miss.
        // At most 4 periodic snapshots to bound overhead. (#3334)
        let snapshot_interval = (self.config.beta_iterations / 4).max(1);
        let mut periodic_snapshots: Vec<GraphBetaState> = Vec::new();

        // Set when the loop bails because the wall-clock deadline was reached
        // (either the between-iteration guard or the inner per-node abort). Drives
        // the post-loop short-circuit that returns the best loop bounds. (#3109)
        let mut hit_deadline = false;

        for iter in 0..self.config.beta_iterations {
            // Deadline check (#3109): bail early if the verification timeout budget
            // is exhausted BETWEEN iterations. Return the current best bounds
            // instead of running all iterations. This is the cheap guard; the
            // inner per-node deadline check below is what usually fires first.
            if self.config.alpha_config.past_deadline() {
                tracing::info!(
                    "Multi-objective β optimization: deadline exceeded at iteration {}/{}, returning best bounds",
                    iter, self.config.beta_iterations
                );
                hit_deadline = true;
                break;
            }

            // Reset gradients
            beta_state.zero_grad();

            // Compute bounds with current β AND capture per-objective A rows in one pass.
            //
            // Deadline granularity (#3109): a single spec-guided backward pass over
            // ~99-199 specs in a deep Conv2d graph can take seconds, and the inner
            // per-node deadline check (constraints/backward/mod.rs) aborts that pass
            // mid-flight with `DeadlineExceeded` once the wall clock crosses the
            // budget. Previously the `?` here propagated that error straight out,
            // DISCARDING every completed beta-opt iteration's bounds. Instead, if we
            // already have a best fully-computed result from an earlier iteration,
            // break and return it via the post-loop deadline short-circuit below.
            // Returning that earlier (valid) spec-guided CROWN bound is sound — we
            // only stop optimizing sooner and yield to BaB/timeout gracefully.
            let (output, node_bounds, intermediate) = match self
                .propagate_crown_with_graph_beta_and_spec_matrix_storing_intermediates(
                    graph,
                    input,
                    context,
                    beta_state,
                    &spec_matrix,
                ) {
                Ok(triple) => triple,
                Err(e) if e.is_deadline_exceeded() && best_loop_result.is_some() => {
                    tracing::info!(
                        "Multi-objective β optimization: inner pass hit deadline at iteration \
                         {}/{}, returning best bounds from a completed iteration",
                        iter,
                        self.config.beta_iterations
                    );
                    hit_deadline = true;
                    break;
                }
                Err(e) => return Err(e),
            };

            let obj_bounds = spec_bounds_to_vec(&output);
            let margin = compute_margin(&obj_bounds);

            // Track best β state by post-hoc margin. See #1694.
            // Also capture the full (bounds, node_bounds) so a mid-loop deadline
            // can return immediately without an extra spec-guided pass. (#3109)
            if margin > best_margin {
                best_margin = margin;
                best_beta_snapshot = Some(beta_state.clone());
                best_loop_result = Some((obj_bounds.clone(), node_bounds));
            }

            // Save periodic snapshot for spec-guided evaluation at end.
            // Post-hoc margin can be β-insensitive, so we save at fixed intervals
            // rather than only when margin improves. (#3334)
            if iter % snapshot_interval == 0 {
                periodic_snapshots.push(beta_state.clone());
            }

            // Compute analytical gradients for the critical objective
            let max_grad = beta_state.compute_analytical_gradients_multi_objective_spec_rows(
                &intermediate,
                &obj_bounds,
                targets.thresholds,
                targets.verified_mask,
                conjunctive,
            );

            // Adam gradient step
            let t = iter + 1;
            beta_state.gradient_step_adam(&self.config.adaptive_config, t);

            // Check convergence
            if max_grad < self.config.beta_tolerance {
                trace!(
                    "Graph β-analytical multi-obj converged at iteration {} (max_grad={:.6})",
                    iter,
                    max_grad
                );
                break;
            }
        }

        // Deadline short-circuit (#3109): if the loop bailed on the wall-clock
        // deadline AND we captured a fully-computed result from a completed
        // iteration, do NOT run the final spec-guided pass or the per-candidate
        // snapshot evaluation below — each is a full spec-guided CROWN pass over
        // every objective, and for deep Conv2d graphs with ~99-199 specs a single
        // pass can itself overrun the remaining budget (and would in fact abort
        // mid-flight with `DeadlineExceeded`, discarding all the beta-opt work).
        // Instead return the best fully-computed bounds captured during the loop.
        // These are valid spec-guided CROWN bounds (sound); we just stop optimizing
        // sooner and yield to BaB/timeout gracefully. We return `None` caches:
        // warm-starting is an optimization, and skipping it is sound (the next
        // round simply recomputes). We also sync `beta_state` to the best snapshot
        // for a consistent (bounds, β) pair.
        let deadline_reached = hit_deadline || self.config.alpha_config.past_deadline();
        if let Some((bounds, node_bounds)) = best_loop_result.filter(|_| deadline_reached) {
            if let Some(best_beta) = best_beta_snapshot {
                *beta_state = best_beta;
            }
            tracing::info!(
                "Multi-objective β optimization: deadline exceeded, returning best \
                 loop bounds without final spec-guided pass or snapshot evaluation"
            );
            let caches = vec![None; targets.objectives.len()];
            return Ok((bounds, node_bounds, caches));
        }
        // Otherwise (no deadline hit, or the deadline hit before any iteration
        // completed so we have no captured result), fall through to compute the
        // final spec-guided bounds so callers always get a valid result. When the
        // deadline already elapsed before iteration 0, this final pass will itself
        // abort with `DeadlineExceeded`, which propagates as a graceful timeout.

        // Compute final spec-guided bounds with the end-of-loop β state.
        let (final_bounds, final_node_bounds, final_cached_las) = self
            .propagate_multi_objective_with_beta_and_cache(
                graph,
                input,
                context,
                beta_state,
                targets,
                seed_caches,
                capture_caches,
            )?;
        let final_margin = compute_margin(&final_bounds);

        // Evaluate all snapshot candidates with spec-guided CROWN and return
        // the β state with the best spec-guided margin. This catches the
        // optimal β that post-hoc tracking may have missed. (#3334, #1694)
        let mut best_overall_margin = final_margin;
        let mut best_overall_bounds = final_bounds;
        let mut best_overall_node_bounds = final_node_bounds;
        let mut best_overall_cached_las = final_cached_las;
        let mut best_overall_beta: Option<GraphBetaState> = None;

        // Deadline guard (#3813): snapshot evaluation runs N spec-guided CROWN
        // passes per candidate (one per objective). For expensive Conv2d graphs
        // with many objectives, this can exceed the remaining timeout budget.
        // Skip snapshot evaluation entirely when past deadline — the final
        // spec-guided bounds above are the best we can return in time.
        if !self.config.alpha_config.past_deadline() {
            let candidates = periodic_snapshots.into_iter().chain(best_beta_snapshot);
            for candidate in candidates {
                // Per-candidate deadline check: bail before each expensive
                // spec-guided evaluation to avoid overrunning the timeout.
                if self.config.alpha_config.past_deadline() {
                    tracing::debug!(
                        "Multi-objective β snapshot evaluation: deadline exceeded, \
                         returning best bounds from {} evaluated candidates",
                        if best_overall_beta.is_some() {
                            "partial"
                        } else {
                            "final-only"
                        }
                    );
                    break;
                }
                let (bounds, node_bounds, cached_las) = self
                    .propagate_multi_objective_with_beta_and_cache(
                        graph,
                        input,
                        context,
                        &candidate,
                        targets,
                        seed_caches,
                        capture_caches,
                    )?;
                let margin = compute_margin(&bounds);
                // Strict > so that ties prefer the end-of-loop β (most recent
                // optimizer state for warm-starting downstream). (#1760)
                if margin > best_overall_margin {
                    best_overall_margin = margin;
                    best_overall_bounds = bounds;
                    best_overall_node_bounds = node_bounds;
                    best_overall_cached_las = cached_las;
                    best_overall_beta = Some(candidate);
                }
            }
        }

        // Update beta_state to match the returned bounds so callers have a
        // consistent (bounds, beta_state) pair for warm-starting. (#1760)
        if let Some(best_beta) = best_overall_beta {
            *beta_state = best_beta;
        }

        Ok((
            best_overall_bounds,
            best_overall_node_bounds,
            best_overall_cached_las,
        ))
    }
}
