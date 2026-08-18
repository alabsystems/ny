// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Post-IBP-prescreen survivor processing for disjunctive multi-clause BaB.
//!
//! Batched multi-spec relaxed clip for `InputClipType::Relaxed` survivors:
//! all children with parent LinearBounds are clipped in a single
//! `batched_relaxed_clip_from_flat` call (N children x T threshold rows),
//! avoiding per-child BoundedTensor construction and per-child clip calls.
//!
//! Per-child fallback for `InputClipType::Complete` or missing parent LinearBounds.
//!
//! Part of #4366 Packet C, #4367 joint multi-spec.

use std::collections::BinaryHeap;
use std::sync::Arc;

use ny_core::{GemmEngine, Result};
use tracing::trace;

use crate::beta_crown::config::InputClipType;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::{certified_affine_sum_f32, LinearBounds, OutwardDirection};
use crate::GraphNetwork;

use super::super::batched_clip::batched_relaxed_clip_from_flat;
use super::super::grouped_semantics::disjunctive_domain_verified;
use super::super::shared::{build_child_input, MultiObjInputDomain};
use super::process_batch::FlatPendingChild;

/// Per-child disposition after batched clip + grouped verification.
enum ChildDisposition {
    /// Clip proved box infeasibility (x_l > x_u on some dimension).
    VerifiedByClip,
    /// Post-clip concretized lower bounds satisfy grouped disjunctive criteria.
    VerifiedByGrouped,
    /// Child survives clip and grouped check; must be queued.
    Survive,
}

/// Batched multi-spec relaxed clip across all surviving children with parent
/// LinearBounds. Instead of N per-child `clip_multi_objective_grouped_safe`
/// calls (each with batch=1), this batches all N children into a single
/// `batched_relaxed_clip_from_flat` call, avoiding:
/// - N individual `build_child_input` reshape+clone cycles before clip
/// - N individual flatten/reshape cycles inside the clip kernel
/// - N separate relaxed_clip_with_infeasible_mask invocations per threshold row
///
/// After batched clipping, performs per-child grouped disjunctive verification
/// using concretized post-clip lower bounds (OR-within-clause, AND-across-clauses).
///
/// Part of #4366 batched child construction + clip, #4367 joint multi-spec.
/// Reference: alpha-beta-CROWN `clip_domains` (`input_split/clip.py:174-274`).
#[allow(clippy::too_many_arguments)]
pub(super) fn push_batched_relaxed_survivors(
    verifier: &BetaCrownVerifier,
    survivors: Vec<FlatPendingChild>,
    shape: &[usize],
    thresholds: &[f32],
    clause_sizes: &[usize],
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_clip: &mut usize,
) -> Result<()> {
    let (with_lb, without_lb): (Vec<_>, Vec<_>) = survivors
        .into_iter()
        .partition(|c| c.linear_bounds.is_some());

    // Batch all children with linear bounds through a single clip pass.
    if !with_lb.is_empty() {
        push_batched_clip_children(
            verifier,
            with_lb,
            shape,
            thresholds,
            clause_sizes,
            queue,
            lifecycle,
            domains_verified_by_clip,
        )?;
    }

    // Children without linear bounds cannot be clipped; push directly.
    for child in without_lb {
        let child_input = build_child_input(&child.flat_lower, &child.flat_upper, shape)?;
        queue.push(MultiObjInputDomain {
            input_bounds: Arc::new(child_input),
            obj_bounds: child.obj_bounds,
            linear_bounds: None,
            depth: child.depth,
            priority: child.priority,
            needs_bounding: true,
            node_bounds_override: None,
            // Parent α slopes for the deferred-rebound warm overlay (step-2C/2D).
            inherited_alpha_state: child.inherited_alpha_state,
        });
    }

    Ok(())
}

/// Core batched clip path: collects flat arrays and LinearBounds references,
/// runs one `batched_relaxed_clip_from_flat` call, then dispatches results.
#[allow(clippy::too_many_arguments)]
fn push_batched_clip_children(
    verifier: &BetaCrownVerifier,
    mut children: Vec<FlatPendingChild>,
    shape: &[usize],
    thresholds: &[f32],
    clause_sizes: &[usize],
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_clip: &mut usize,
) -> Result<()> {
    let n = children.len();

    // The stored f32 coefficients are certified only to within their carried
    // error envelope (#vnncomp-aw-soundness). Discharge it into the bias over
    // each child's own box before the coefficients drive the batched clip and
    // the grouped disjunctive check: rows whose penalty is non-finite degrade
    // to a ±inf bias, which can never verify and never clips.
    for child in &mut children {
        if let Some(lb) = child.linear_bounds.as_mut() {
            if lb.has_coeff_err() {
                match (child.flat_lower.as_slice(), child.flat_upper.as_slice()) {
                    (Some(in_l), Some(in_u)) => lb.fold_coeff_err_into_bias(in_l, in_u),
                    // Non-contiguous flats cannot be mapped onto the
                    // coefficient columns; degrade rather than assume.
                    _ => lb.discharge_coeff_err_to_conservative(),
                }
            }
        }
    }

    // Collect flat arrays and LinearBounds references for the batched clip call.
    let flat_lowers: Vec<_> = children.iter().map(|c| c.flat_lower.clone()).collect();
    let flat_uppers: Vec<_> = children.iter().map(|c| c.flat_upper.clone()).collect();

    let lb_refs: Vec<&LinearBounds> = children
        .iter()
        .map(|c| {
            c.linear_bounds.as_ref().ok_or_else(|| {
                ny_core::NyError::InternalError(
                    "partitioned with Some but linear_bounds is None".into(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Single batched clip call: N children x T threshold rows.
    // This replaces N individual clip_multi_objective_grouped_safe calls.
    // Clause-aware (#disj-cross-clause-clip-unsat): per-clause intersection +
    // union bbox — see `batched_relaxed_clip_core_grouped`.
    let clip_result = batched_relaxed_clip_from_flat(
        &flat_lowers,
        &flat_uppers,
        &lb_refs,
        thresholds,
        clause_sizes,
        verifier.config.verify_upper_bound,
        verifier.config.relaxed_clip_iterations,
    )?;

    // Phase 1: compute per-child disposition while children is still borrowed.
    // lb_refs borrows from children, so we cannot consume children yet.
    let dispositions: Vec<ChildDisposition> = {
        let mut d = Vec::with_capacity(n);
        for i in 0..n {
            if clip_result.verified[i] {
                d.push(ChildDisposition::VerifiedByClip);
                continue;
            }

            let postclip_obj_bounds = concretize_postclip_lower_bounds(
                &clip_result.clipped_lowers[i],
                &clip_result.clipped_uppers[i],
                lb_refs[i],
                thresholds,
                verifier.config.verify_upper_bound,
            );
            if disjunctive_domain_verified(&postclip_obj_bounds, thresholds, clause_sizes) {
                d.push(ChildDisposition::VerifiedByGrouped);
                continue;
            }

            d.push(ChildDisposition::Survive);
        }
        d
    };

    // Phase 2: consume children now that lb_refs is no longer needed.
    drop(lb_refs);
    for (i, child) in children.into_iter().enumerate() {
        match dispositions[i] {
            ChildDisposition::VerifiedByClip => {
                lifecycle.domains_verified += 1;
                *domains_verified_by_clip += 1;
                trace!("batched clip: child {} verified by box infeasibility", i);
            }
            ChildDisposition::VerifiedByGrouped => {
                lifecycle.domains_verified += 1;
                *domains_verified_by_clip += 1;
                trace!(
                    "batched clip: child {} verified by grouped disjunctive check",
                    i
                );
            }
            ChildDisposition::Survive => {
                // Materialize BoundedTensor only for survivors (deferred from pre-clip).
                let child_input = build_child_input(
                    &clip_result.clipped_lowers[i],
                    &clip_result.clipped_uppers[i],
                    shape,
                )?;
                queue.push(MultiObjInputDomain {
                    input_bounds: Arc::new(child_input),
                    obj_bounds: child.obj_bounds,
                    linear_bounds: None,
                    depth: child.depth,
                    priority: child.priority,
                    needs_bounding: true,
                    node_bounds_override: None,
                    // Parent α slopes for the deferred-rebound warm overlay
                    // (step-2C/2D).
                    inherited_alpha_state: child.inherited_alpha_state,
                });
            }
        }
    }

    trace!("batched clip: processed {} children in single pass", n);
    Ok(())
}

/// Concretize post-clip lower bounds for grouped disjunctive verification.
///
/// For each threshold row, computes the sound lower bound of
/// `coeffs[row] . x + bias[row]` over the clipped box `[x_l, x_u]`.
///
/// Sound lower bound formula: for each dimension d,
///   if a_d >= 0: use x_l_d (minimum of a_d * x_d)
///   if a_d < 0:  use x_u_d (minimum of a_d * x_d)
///
/// Uses f64 accumulation and a directed `next_down_f32` cast, matching
/// `concretize_bounds` in `relaxed_clip.rs` (#2303). Results are paired with
/// `f32::INFINITY` (unknown upper) for the grouped disjunctive verification
/// check.
///
/// Reads the raw coefficient rows, so any certified coefficient error must
/// have been discharged (`fold_coeff_err_into_bias`) before this is called;
/// bounds still carrying error degrade to `-inf` (never verify).
///
/// Reference: `clip_multi_objective_grouped_safe` concretization path.
pub(super) fn concretize_postclip_lower_bounds(
    clipped_lower: &ndarray::ArrayD<f32>,
    clipped_upper: &ndarray::ArrayD<f32>,
    linear_bounds: &LinearBounds,
    thresholds: &[f32],
    verify_upper_bound: bool,
) -> Vec<(f32, f32)> {
    let x_dim = clipped_lower.len();
    let n_rows = thresholds.len().min(linear_bounds.lower_a().nrows());

    if linear_bounds.has_coeff_err() {
        return vec![(f32::NEG_INFINITY, f32::INFINITY); n_rows];
    }

    let mut result = Vec::with_capacity(n_rows);
    for row_idx in 0..n_rows {
        let (coeffs_row, bias) = if verify_upper_bound {
            let row = linear_bounds.upper_a().row(row_idx);
            let b = -linear_bounds.upper_b()[row_idx];
            let negated: Vec<f32> = row.iter().map(|v| -v).collect();
            (negated, b)
        } else {
            let row = linear_bounds.lower_a().row(row_idx);
            let b = linear_bounds.lower_b()[row_idx];
            (row.to_vec(), b)
        };

        // Sound lower bound: use x_l when coeff positive, x_u when negative.
        let lb_val = certified_affine_sum_f32(
            bias,
            (0..x_dim.min(coeffs_row.len())).map(|d| {
                let a = coeffs_row[d];
                let endpoint = if a >= 0.0 {
                    clipped_lower[[d]]
                } else {
                    clipped_upper[[d]]
                };
                (a, endpoint)
            }),
            OutwardDirection::Lower,
        );

        // Each f64 addition above and the final f64→f32 cast round DOWN:
        // a verdict-bearing lower bound cannot rely on final-only widening
        // after cancellation. NaN degrades to -inf (never verifies).
        let lb_f32 = if lb_val.is_nan() {
            f32::NEG_INFINITY
        } else {
            ny_tensor::next_down_f32(lb_val as f32)
        };
        result.push((lb_f32, f32::INFINITY));
    }

    result
}

/// Fallback per-child clip for InputClipType::Complete or disabled clip.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_fallback_survivors(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    survivors: Vec<FlatPendingChild>,
    shape: &[usize],
    thresholds: &[f32],
    engine: Option<&dyn GemmEngine>,
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    _lifecycle: &mut GraphBabLifecycle,
    _domains_verified_by_clip: &mut usize,
) -> Result<()> {
    for mut child in survivors {
        let mut child_input = build_child_input(&child.flat_lower, &child.flat_upper, shape)?;
        let mut complete_clip_node_bounds = None;
        if verifier.config.enable_relaxed_clip {
            if let Some(linear_bounds) = child.linear_bounds.as_mut() {
                // Same envelope discharge as the batched lane: the complete
                // clip consumes the raw coefficient rows for box tightening.
                BetaCrownVerifier::discharge_coeff_err_for_clip(linear_bounds, &child_input);
                let bt_shape = child_input.lower().shape().to_vec();
                if matches!(verifier.config.input_clip_type, InputClipType::Complete) {
                    let clip_outcome = verifier.complete_clip_with_precomputed_specs(
                        &child_input,
                        &bt_shape,
                        linear_bounds,
                        thresholds,
                    )?;
                    child_input = clip_outcome.bounds;
                    complete_clip_node_bounds =
                        match super::super::super::clip_complete::build_graph_complete_clip_node_bounds(
                            graph,
                            &child_input,
                            linear_bounds,
                            thresholds,
                            verifier.config.verify_upper_bound,
                            verifier.config.clip_neuron_selection_ratio,
                            engine,
                        ) {
                            Ok(nb) => nb,
                            Err(err) => {
                                trace!("graph complete clip: skipping due to {}", err);
                                None
                            }
                        };
                }
            }
        }
        queue.push(MultiObjInputDomain {
            input_bounds: Arc::new(child_input),
            obj_bounds: child.obj_bounds,
            linear_bounds: None,
            depth: child.depth,
            priority: child.priority,
            needs_bounding: true,
            node_bounds_override: complete_clip_node_bounds.map(Arc::new),
            // Parent α slopes for the deferred-rebound warm overlay (step-2C/2D).
            inherited_alpha_state: child.inherited_alpha_state,
        });
    }
    Ok(())
}
