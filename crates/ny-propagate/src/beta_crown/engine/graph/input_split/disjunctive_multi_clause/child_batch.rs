// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-child-batch (S1): consolidated child pipeline for the
//! reorder-prescreen disjunctive lane.
//!
//! The historical path (`process_reorder_prescreen_batch`) materializes every
//! child as a `FlatPendingChild` carrying per-child deep clones: two flat
//! `ArrayD` box clones, an `obj_bounds` clone, and a FULL parent
//! `LinearBounds` clone per child — then RE-clones the flats for the IBP
//! prescreen stack and a third time for the batched relaxed clip. On the
//! lsnc-class nets (D=6, S=39) this per-child malloc churn is a dominant
//! serial cost (`docs/LSNC_BATCH_TENSOR_DESIGN.md` S1).
//!
//! This module replaces the chain with ONE `ChildBatch`: contiguous
//! `(2B, D)` child rows written once at split time, a `parent_idx` map, and
//! per-PARENT retained `obj_bounds`/`LinearBounds` (both children of one
//! parent share the parent's planes verbatim). The prescreen consumes the
//! stacked rows directly (`batched_ibp_prescreen_from_stacked`), the clip
//! gathers coefficient rows from the parent planes via `parent_idx`
//! (`batched_relaxed_clip_from_stacked`), and per-child `LinearBounds` clones
//! happen ONLY where the math genuinely requires a per-child mutation: the
//! coefficient-error discharge over each child's own box (I-A10 — never fold
//! on the shared parent planes).
//!
//! BIT-PARITY CLASS: pure data movement — every kernel invoked
//! (`run_ibp_forward`, `relaxed_clip_with_infeasible_mask`,
//! `fold_coeff_err_into_bias`, `concretize_postclip_lower_bounds`) is the
//! UNCHANGED production kernel fed bit-identical inputs, and queue pushes
//! happen in the identical order (with-LinearBounds survivors first, then
//! without, both in pending order — heap tie-breaking is part of the parity
//! criterion). Parity: `test_child_batch_reorder_prescreen_parity_lsnc_s1`.

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Duration;

use ndarray::{Array2, ArrayD, Axis, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use tracing::trace;

use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::bounds::LinearBounds;
use crate::GraphNetwork;

use super::super::batched_clip::{
    batched_relaxed_clip_from_planes, batched_relaxed_clip_from_stacked,
    concretize_postclip_lower_bounds_planes, ParentClipPlane,
};
use super::super::grouped_semantics::disjunctive_domain_verified;
use super::super::ibp_prescreen_flat::batched_ibp_prescreen_from_stacked;
use super::super::shared::{build_child_input_owned, MultiObjInputDomain};
use super::process_batch::{
    edge_domain_rows, graph_ibp_prescreen_error_should_skip, try_edge_alpha_pass,
    try_edge_milp_escalation,
};
use super::push_survivors::concretize_postclip_lower_bounds;

/// #lsnc-child-batch (S1) gate. Default ON (flipped after the parity test +
/// the end-to-end lsnc verdict-identity A/B ran green: instances 0/1/3/5/6
/// verdicts and per-batch verified/clipped/gap trajectories identical,
/// 2026-07-18; measured e2e ~1.03x on a full instance-3 proof, logged-batch
/// dps +0.6-5.1%); set `NY_INPUT_SPLIT_CHILD_BATCH=0` to force the
/// historical `FlatPendingChild` path, which stays in-tree as the A/B +
/// parity reference. Applies ONLY to the reorder-prescreen lane with
/// `InputClipType::Relaxed` + `enable_relaxed_clip` (the lsnc preset); every
/// other configuration takes the unchanged reference path.
static CHILD_BATCH_MODE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// Whether the consolidated child pipeline is enabled (see [`CHILD_BATCH_MODE`]).
pub(super) fn input_split_child_batch_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match CHILD_BATCH_MODE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = !matches!(
                std::env::var("NY_INPUT_SPLIT_CHILD_BATCH").ok().as_deref(),
                Some("0") | Some("false")
            );
            CHILD_BATCH_MODE.store(i8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// Test-only runtime override for the child-batch gate: `Some(true|false)`
/// forces ON/OFF, `None` restores the env-derived default (mirrors
/// `force_batched_relu`). Tests MUST restore `None` afterward.
#[cfg(test)]
pub(crate) fn force_child_batch(mode: Option<bool>) {
    use std::sync::atomic::Ordering;
    let v = match mode {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    CHILD_BATCH_MODE.store(v, Ordering::Relaxed);
}

/// #lsnc-clip-planes (S5) gate. Default ON (flipped after the parity suite +
/// the end-to-end lsnc verdict-identity A/B ran green on 2026-07-18:
/// instances 0/1/3/5 verdicts identical, matched-batch
/// popped/verified/clipped counters identical on 3/5/6; measured
/// `split_screen_s` 0.65-0.94 -> 0.04-0.06 on the big batches, dps
/// x2.0-2.2, walls x1.6; fragility watch 24/30/32/54/70 all unsat with
/// ~7 s added margin); set `NY_INPUT_SPLIT_CLIP_PLANES=0` to force the S1
/// stacked reference, which stays in-tree as the A/B + parity baseline.
/// The planes-based clip for the child-batch relaxed-clip lane:
/// per-threshold coefficient gathers from the SHARED parent planes via
/// `parent_idx` (no per-child `LinearBounds` clones), the batched used-side
/// coefficient-error bias fold, the caller-scratch `n_spec = 1` per-row clip
/// core with fixed-point elision (`relaxed_clip_single_spec_row_fast`), and
/// the planes-based post-clip concretize. BIT-PARITY CLASS vs the S1 stacked
/// path (`test_clip_planes_reorder_prescreen_parity_lsnc_s5`,
/// `test_batched_clip_planes_matches_stacked_s5`); any unexpected plane shape
/// or layout DECLINES to the unchanged S1 reference body (fail-closed).
static CLIP_PLANES_MODE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// Whether the S5 planes-based clip is enabled (see [`CLIP_PLANES_MODE`]).
fn input_split_clip_planes_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match CLIP_PLANES_MODE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = !matches!(
                std::env::var("NY_INPUT_SPLIT_CLIP_PLANES").ok().as_deref(),
                Some("0") | Some("false")
            );
            CLIP_PLANES_MODE.store(i8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// Test-only probe: number of survivor groups HANDLED by the S5 planes path
/// (a declined batch does not count). Parity tests assert it advanced so a
/// silent decline cannot make the comparison vacuous.
#[cfg(test)]
pub(crate) static CLIP_PLANES_HANDLED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Test-only runtime override for the S5 clip-planes gate (mirrors
/// [`force_child_batch`]). Tests MUST hold
/// `propagation::batched::SPEC_GATE_TEST_LOCK` while forcing and restore
/// `None` afterward.
#[cfg(test)]
pub(crate) fn force_clip_planes(mode: Option<bool>) {
    use std::sync::atomic::Ordering;
    let v = match mode {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    CLIP_PLANES_MODE.store(v, Ordering::Relaxed);
}

/// Per-parent retained state: both children of one parent share these
/// verbatim (the historical path deep-cloned them into each child).
struct ChildBatchParent {
    obj_bounds: Vec<(f32, f32)>,
    linear_bounds: Option<LinearBounds>,
    /// Depth of the CHILDREN (parent depth + 1).
    child_depth: usize,
    priority: f32,
    /// Parent's refined α slopes for the deferred-rebound warm-α overlay
    /// (cgan step-2C/2D). `None` (the default gate off) keeps the frozen path.
    inherited_alpha_state: Option<Arc<crate::bounds::GraphAlphaState>>,
}

/// #lsnc-child-batch (S1) fast path for `process_reorder_prescreen_batch`.
/// Same per-domain screening loop (timeout / domain-limit / verified /
/// edge-α / edge-MILP / depth / SB scoring), but children are written as
/// contiguous rows of one batch instead of per-child heap objects.
#[allow(clippy::too_many_arguments)]
pub(super) fn process_reorder_prescreen_child_batch(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domains: Vec<MultiObjInputDomain>,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn GemmEngine>,
    bab_timeout: Duration,
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_clip: &mut usize,
) -> Result<Option<BetaCrownResult>> {
    // The ChildBatch arenas: contiguous (2B, D) child rows + parent records.
    let mut lower_data: Vec<f32> = Vec::new();
    let mut upper_data: Vec<f32> = Vec::new();
    let mut parent_idx: Vec<usize> = Vec::new();
    let mut parents: Vec<ChildBatchParent> = Vec::new();
    let mut child_shape: Option<Vec<usize>> = None;
    let mut x_dim = 0usize;
    // #relational-bab option B: per-wave α edge-pass budget.
    let mut edge_alpha_budget = verifier.config.input_split_edge_alpha_top;

    for mut domain in domains {
        if lifecycle.start_time.elapsed() > bab_timeout {
            return Ok(Some(lifecycle.timeout_result()));
        }
        if lifecycle.domains_explored >= verifier.config.max_domains {
            return Ok(Some(lifecycle.build_result(
                BabVerificationStatus::Unknown {
                    reason: format!(
                        "Domain limit {}: {}/{} verified",
                        verifier.config.max_domains,
                        lifecycle.domains_verified,
                        lifecycle.domains_explored
                    ),
                },
            )));
        }

        lifecycle.domains_explored += 1;
        lifecycle.max_depth_reached = lifecycle.max_depth_reached.max(domain.depth);

        if lifecycle.domains_explored.is_multiple_of(1000) || lifecycle.domains_explored <= 5 {
            trace!(
                "[disjunctive-multi-clause] explored={} verified={} clipped={} depth={} queue={} pri={:.4}",
                lifecycle.domains_explored,
                lifecycle.domains_verified,
                *domains_verified_by_clip,
                domain.depth,
                queue.len(),
                domain.priority,
            );
        }

        if disjunctive_domain_verified(&domain.obj_bounds, thresholds, clause_sizes) {
            lifecycle.domains_verified += 1;
            continue;
        }
        // #relational-bab option B (config-gated, default inert): identical to
        // the reference loop — see `process_reorder_prescreen_batch`.
        if verifier.config.input_split_edge_alpha && edge_alpha_budget > 0 {
            if let Some(row_indices) = edge_domain_rows(
                &domain,
                thresholds,
                clause_sizes,
                verifier.config.input_split_edge_milp_gap,
                verifier.config.input_split_edge_milp_depth,
                spec_matrix.nrows(),
            ) {
                edge_alpha_budget -= 1;
                if let Some(fresh) = try_edge_alpha_pass(
                    verifier,
                    graph,
                    &domain,
                    spec_matrix,
                    &row_indices,
                    engine,
                    lifecycle.start_time + bab_timeout,
                ) {
                    domain.obj_bounds =
                        super::super::batching::tighten_obj_lower_bounds(&domain.obj_bounds, fresh);
                    if disjunctive_domain_verified(&domain.obj_bounds, thresholds, clause_sizes) {
                        lifecycle.domains_verified += 1;
                        continue;
                    }
                }
            }
        }
        // #relational-bab EDGE-DOMAIN ESCALATION: identical to the reference.
        if try_edge_milp_escalation(
            verifier,
            graph,
            &domain,
            spec_matrix,
            thresholds,
            clause_sizes,
            engine,
            lifecycle.start_time + bab_timeout,
        ) {
            lifecycle.domains_verified += 1;
            continue;
        }
        if domain.depth >= verifier.config.max_depth {
            lifecycle.unresolved_due_to_depth = true;
            continue;
        }

        let domain_bounds: Vec<f32> = domain
            .obj_bounds
            .iter()
            .map(|(lower, upper)| {
                if verifier.config.verify_upper_bound {
                    *upper
                } else {
                    *lower
                }
            })
            .collect();
        let split_dim = verifier.select_input_dimension_sb(
            domain.input_bounds.as_ref(),
            domain.linear_bounds.as_ref(),
            Some(domain_bounds.as_slice()),
            Some(thresholds),
        );
        let flat = domain.input_bounds.as_ref().flatten();

        if split_dim >= flat.len() {
            lifecycle.unresolved_due_to_unsplittable = true;
            continue;
        }
        let l = flat.lower()[[split_dim]];
        let u = flat.upper()[[split_dim]];
        if !l.is_finite() || !u.is_finite() || u <= l {
            lifecycle.unresolved_due_to_unsplittable = true;
            continue;
        }

        let mid = l + (u - l) / 2.0;
        let shape = domain.input_bounds.as_ref().lower().shape().to_vec();
        if child_shape.is_none() {
            child_shape = Some(shape);
            x_dim = flat.len();
        }
        if flat.len() != x_dim {
            // All domains in a batch share the same input shape; a mismatch
            // would corrupt the row arena. The reference path surfaces the
            // same class of error at its prescreen stack.
            return Err(NyError::InvalidSpec(format!(
                "child batch: domain flat dim {} != batch dim {}",
                flat.len(),
                x_dim
            )));
        }

        // Retain per-parent state ONCE (the reference cloned it per child).
        let p = parents.len();
        parents.push(ChildBatchParent {
            obj_bounds: std::mem::take(&mut domain.obj_bounds),
            linear_bounds: domain.linear_bounds.take(),
            child_depth: domain.depth + 1,
            priority: domain.priority,
            inherited_alpha_state: domain.inherited_alpha_state.take(),
        });

        // Left child: [l, mid] on split_dim — one contiguous row write.
        let row = lower_data.len();
        lower_data.extend(flat.lower().iter());
        upper_data.extend(flat.upper().iter());
        upper_data[row + split_dim] = mid;
        parent_idx.push(p);

        // Right child: [mid, u] on split_dim.
        let row = lower_data.len();
        lower_data.extend(flat.lower().iter());
        upper_data.extend(flat.upper().iter());
        lower_data[row + split_dim] = mid;
        parent_idx.push(p);
    }

    if parents.is_empty() {
        return Ok(None);
    }

    let n = parent_idx.len();
    let shape = child_shape.ok_or_else(|| {
        NyError::InvalidSpec("no child shape recorded despite non-empty child batch".to_string())
    })?;

    // Batched IBP prescreen straight off the stacked rows (no per-child
    // clones, no re-stack). Same error-skip contract as the reference.
    let mut stacked_shape = Vec::with_capacity(1 + shape.len());
    stacked_shape.push(n);
    stacked_shape.extend_from_slice(&shape);
    let stacked_lower = ArrayD::from_shape_vec(IxDyn(&stacked_shape), lower_data.clone())
        .map_err(|e| NyError::InvalidSpec(format!("child batch: reshape stacked lower: {}", e)))?;
    let stacked_upper = ArrayD::from_shape_vec(IxDyn(&stacked_shape), upper_data.clone())
        .map_err(|e| NyError::InvalidSpec(format!("child batch: reshape stacked upper: {}", e)))?;
    let verified_mask = match batched_ibp_prescreen_from_stacked(
        graph,
        stacked_lower,
        stacked_upper,
        n,
        spec_matrix,
        thresholds,
        Some(clause_sizes),
        false,
        engine,
    ) {
        Ok(mask) => mask,
        Err(err) if graph_ibp_prescreen_error_should_skip(&err) => {
            tracing::debug!(
                "disjunctive reorder prescreen: skipping enhancement-only IBP pass for {} children after {}",
                n,
                err
            );
            vec![false; n]
        }
        Err(err) => return Err(err),
    };

    // Partition survivors, preserving pending order within each group —
    // with-LinearBounds children push first (after clip), then without
    // (matching the reference `partition` + push order exactly: heap
    // tie-breaking is part of the parity criterion).
    let mut with_lb: Vec<usize> = Vec::new();
    let mut without_lb: Vec<usize> = Vec::new();
    for (i, verified) in verified_mask.iter().enumerate() {
        if *verified {
            lifecycle.domains_verified += 1;
        } else if parents[parent_idx[i]].linear_bounds.is_some() {
            with_lb.push(i);
        } else {
            without_lb.push(i);
        }
    }

    if !with_lb.is_empty() {
        push_child_batch_clip_survivors(
            verifier,
            &with_lb,
            &lower_data,
            &upper_data,
            x_dim,
            &parent_idx,
            &parents,
            &shape,
            thresholds,
            clause_sizes,
            queue,
            lifecycle,
            domains_verified_by_clip,
        )?;
    }

    // Children without parent LinearBounds cannot be clipped; push directly
    // (reference: `push_batched_relaxed_survivors` without_lb loop).
    for &ci in &without_lb {
        let parent = &parents[parent_idx[ci]];
        let (row_l, row_u) = child_row_arrays(&lower_data, &upper_data, ci, x_dim)?;
        let child_input = build_child_input_owned(row_l, row_u, &shape)?;
        queue.push(MultiObjInputDomain {
            input_bounds: Arc::new(child_input),
            obj_bounds: parent.obj_bounds.clone(),
            linear_bounds: None,
            depth: parent.child_depth,
            priority: parent.priority,
            needs_bounding: true,
            node_bounds_override: None,
            // Parent α slopes for the deferred-rebound warm overlay (step-2C/2D).
            inherited_alpha_state: parent.inherited_alpha_state.clone(),
        });
    }

    Ok(None)
}

/// Extract child `ci`'s flat row pair as owned 1-D arrays.
fn child_row_arrays(
    lower_data: &[f32],
    upper_data: &[f32],
    ci: usize,
    x_dim: usize,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    let s = ci * x_dim;
    let row_l = ArrayD::from_shape_vec(IxDyn(&[x_dim]), lower_data[s..s + x_dim].to_vec())
        .map_err(|e| NyError::InvalidSpec(format!("child batch: row lower[{}]: {}", ci, e)))?;
    let row_u = ArrayD::from_shape_vec(IxDyn(&[x_dim]), upper_data[s..s + x_dim].to_vec())
        .map_err(|e| NyError::InvalidSpec(format!("child batch: row upper[{}]: {}", ci, e)))?;
    Ok((row_l, row_u))
}

/// Clip + disposition for the with-LinearBounds survivor group. Mirrors
/// `push_batched_clip_children` with coefficient rows gathered from the
/// SHARED parent planes; a per-child `LinearBounds` clone happens only for
/// the coefficient-error discharge over that child's own box (I-A10).
#[allow(clippy::too_many_arguments)]
fn push_child_batch_clip_survivors(
    verifier: &BetaCrownVerifier,
    with_lb: &[usize],
    lower_data: &[f32],
    upper_data: &[f32],
    x_dim: usize,
    parent_idx: &[usize],
    parents: &[ChildBatchParent],
    shape: &[usize],
    thresholds: &[f32],
    clause_sizes: &[usize],
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_clip: &mut usize,
) -> Result<()> {
    // #lsnc-clip-planes (S5): the planes-based fast path (opt-in, fail-closed
    // decline to the unchanged reference body below).
    if input_split_clip_planes_enabled()
        && try_push_child_batch_clip_survivors_planes(
            verifier,
            with_lb,
            lower_data,
            upper_data,
            x_dim,
            parent_idx,
            parents,
            shape,
            thresholds,
            clause_sizes,
            queue,
            lifecycle,
            domains_verified_by_clip,
        )?
    {
        return Ok(());
    }

    let m = with_lb.len();

    // Compact survivor rows into the (m, x_dim) pre-clip originals.
    let mut orig_l_data = Vec::with_capacity(m * x_dim);
    let mut orig_u_data = Vec::with_capacity(m * x_dim);
    for &ci in with_lb {
        let s = ci * x_dim;
        orig_l_data.extend_from_slice(&lower_data[s..s + x_dim]);
        orig_u_data.extend_from_slice(&upper_data[s..s + x_dim]);
    }
    let orig_l = ArrayD::from_shape_vec(IxDyn(&[m, x_dim]), orig_l_data)
        .map_err(|e| NyError::InvalidSpec(format!("child batch: reshape orig_l: {}", e)))?;
    let orig_u = ArrayD::from_shape_vec(IxDyn(&[m, x_dim]), orig_u_data)
        .map_err(|e| NyError::InvalidSpec(format!("child batch: reshape orig_u: {}", e)))?;

    // Coefficient-error discharge BEFORE clip use (I-A10): rows carrying
    // certified error fold it into the bias over EACH CHILD'S OWN box —
    // cloned per child so the shared parent planes are never mutated
    // (shared-vs-owned aliasing hazard). Children whose parent planes carry
    // no error share the parent's `LinearBounds` by reference (the reference
    // path's per-child clone was byte-identical to the parent in that case).
    let mut folded: Vec<Option<LinearBounds>> = Vec::with_capacity(m);
    for (k, &ci) in with_lb.iter().enumerate() {
        let lb = parents[parent_idx[ci]]
            .linear_bounds
            .as_ref()
            .ok_or_else(|| {
                NyError::InternalError(
                    "child batch: with_lb child has no parent linear_bounds".into(),
                )
            })?;
        if lb.has_coeff_err() {
            let mut owned = lb.clone();
            let row_l = orig_l.index_axis(Axis(0), k);
            let row_u = orig_u.index_axis(Axis(0), k);
            match (row_l.as_slice(), row_u.as_slice()) {
                (Some(in_l), Some(in_u)) => owned.fold_coeff_err_into_bias(in_l, in_u),
                // Non-contiguous rows cannot be mapped onto the coefficient
                // columns; degrade rather than assume (same as the reference).
                _ => owned.discharge_coeff_err_to_conservative(),
            }
            folded.push(Some(owned));
        } else {
            folded.push(None);
        }
    }
    let mut lb_refs: Vec<&LinearBounds> = Vec::with_capacity(m);
    for (k, &ci) in with_lb.iter().enumerate() {
        let shared = parents[parent_idx[ci]]
            .linear_bounds
            .as_ref()
            .ok_or_else(|| {
                NyError::InternalError(
                    "child batch: with_lb child has no parent linear_bounds".into(),
                )
            })?;
        lb_refs.push(folded[k].as_ref().unwrap_or(shared));
    }

    // Single batched clip call: m children x T threshold rows, coefficients
    // gathered from the shared/folded planes. Clause-aware
    // (#disj-cross-clause-clip-unsat): per-clause intersection + union bbox.
    let (clip_l, clip_u, verified_clip) = batched_relaxed_clip_from_stacked(
        &orig_l,
        &orig_u,
        &lb_refs,
        thresholds,
        clause_sizes,
        verifier.config.verify_upper_bound,
        verifier.config.relaxed_clip_iterations,
    )?;

    for (k, &ci) in with_lb.iter().enumerate() {
        let parent = &parents[parent_idx[ci]];
        if verified_clip[k] {
            lifecycle.domains_verified += 1;
            *domains_verified_by_clip += 1;
            trace!("batched clip: child {} verified by box infeasibility", k);
            continue;
        }

        let row_l = ArrayD::from_shape_vec(
            IxDyn(&[x_dim]),
            (0..x_dim).map(|d| clip_l[[k, d]]).collect(),
        )
        .map_err(|e| NyError::InvalidSpec(format!("child batch: clipped lower[{}]: {}", k, e)))?;
        let row_u = ArrayD::from_shape_vec(
            IxDyn(&[x_dim]),
            (0..x_dim).map(|d| clip_u[[k, d]]).collect(),
        )
        .map_err(|e| NyError::InvalidSpec(format!("child batch: clipped upper[{}]: {}", k, e)))?;

        let postclip_obj_bounds = concretize_postclip_lower_bounds(
            &row_l,
            &row_u,
            lb_refs[k],
            thresholds,
            verifier.config.verify_upper_bound,
        );
        if disjunctive_domain_verified(&postclip_obj_bounds, thresholds, clause_sizes) {
            lifecycle.domains_verified += 1;
            *domains_verified_by_clip += 1;
            trace!(
                "batched clip: child {} verified by grouped disjunctive check",
                k
            );
            continue;
        }

        let child_input = build_child_input_owned(row_l, row_u, shape)?;
        queue.push(MultiObjInputDomain {
            input_bounds: Arc::new(child_input),
            obj_bounds: parent.obj_bounds.clone(),
            linear_bounds: None,
            depth: parent.child_depth,
            priority: parent.priority,
            needs_bounding: true,
            node_bounds_override: None,
            // Parent α slopes for the deferred-rebound warm overlay (step-2C/2D).
            inherited_alpha_state: parent.inherited_alpha_state.clone(),
        });
    }

    trace!("batched clip: processed {} children in single pass", m);
    Ok(())
}

/// #lsnc-clip-planes (S5): planes-based clip + disposition for the
/// with-LinearBounds survivor group. Consumer-surface identical to
/// [`push_child_batch_clip_survivors`]'s reference body (identical lifecycle
/// counters, dispositions, and queue pushes in identical order, with
/// bit-identical boxes) while eliminating the remaining per-child /
/// per-threshold rebuild work the profile shows dominating `split_screen_s`:
///
/// * per-child `LinearBounds` clone+fold → ONE batched used-side bias fold
///   over the shared parent err planes (`[m, S]` biases; identical per-row
///   expressions to `fold_coeff_err_into_bias`, evaluated over each child's
///   OWN box — I-A10; the shared parent planes are never mutated);
/// * per-threshold `(N, 1, x_dim)` `Array3` rebuild + fresh clip-call
///   allocation cycle → parent-plane gathers into reused scratch + the
///   caller-scratch `n_spec = 1` row core with per-entry fixed-point elision
///   (`batched_relaxed_clip_from_planes`);
/// * per-child post-clip `ArrayD` row rebuilds + `Vec` allocations →
///   planes-based concretize into one reused buffer
///   (`concretize_postclip_lower_bounds_planes`).
///
/// Returns `Ok(true)` when handled. Returns `Ok(false)` — BEFORE any
/// lifecycle/queue mutation — to DECLINE on any unexpected plane geometry
/// (non-contiguous coefficient storage, column/width mismatch, err-plane
/// shape surprise); the caller then runs the unchanged reference body, which
/// reproduces the historical behavior for that batch (fail-closed).
/// Parity: `test_clip_planes_reorder_prescreen_parity_lsnc_s5`,
/// `test_batched_clip_planes_matches_stacked_s5`.
#[allow(clippy::too_many_arguments)]
fn try_push_child_batch_clip_survivors_planes(
    verifier: &BetaCrownVerifier,
    with_lb: &[usize],
    lower_data: &[f32],
    upper_data: &[f32],
    x_dim: usize,
    parent_idx: &[usize],
    parents: &[ChildBatchParent],
    shape: &[usize],
    thresholds: &[f32],
    clause_sizes: &[usize],
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_clip: &mut usize,
) -> Result<bool> {
    use std::borrow::Cow;

    let m = with_lb.len();
    let n_thr = thresholds.len();
    let verify_upper = verifier.config.verify_upper_bound;

    // Per-UNIQUE-parent used-side planes (both children of a parent share the
    // plane; the reference re-gathered the identical row per child per
    // threshold). All decline checks happen here, before any state mutation.
    let mut parent_slot: Vec<Option<usize>> = vec![None; parents.len()];
    let mut planes: Vec<ParentClipPlane<'_>> = Vec::new();
    let mut plane_src: Vec<&LinearBounds> = Vec::new();
    let mut child_plane: Vec<usize> = Vec::with_capacity(m);
    for &ci in with_lb {
        let p = parent_idx[ci];
        let slot = match parent_slot[p] {
            Some(slot) => slot,
            None => {
                let Some(lb) = parents[p].linear_bounds.as_ref() else {
                    return Ok(false);
                };
                let used_a = if verify_upper {
                    lb.upper_a()
                } else {
                    lb.lower_a()
                };
                let nrows = used_a.nrows();
                if used_a.ncols() != x_dim {
                    return Ok(false);
                }
                let Some(flat) = used_a.as_slice() else {
                    return Ok(false);
                };
                let used_b = if verify_upper {
                    lb.upper_b()
                } else {
                    lb.lower_b()
                };
                if used_b.len() != nrows {
                    return Ok(false);
                }
                let used_err = if verify_upper {
                    lb.upper_a_err()
                } else {
                    lb.lower_a_err()
                };
                if let Some(err) = used_err {
                    if err.shape() != used_a.shape() {
                        return Ok(false);
                    }
                }
                let coeffs: Cow<'_, [f32]> = if verify_upper {
                    // Clip sign convention: negate upper rows (matches
                    // `build_batched_coefficients`), once per parent.
                    Cow::Owned(flat.iter().map(|v| -v).collect())
                } else {
                    Cow::Borrowed(flat)
                };
                planes.push(ParentClipPlane { coeffs, nrows });
                plane_src.push(lb);
                let slot = planes.len() - 1;
                parent_slot[p] = Some(slot);
                slot
            }
        };
        child_plane.push(slot);
    }

    // Compact survivor rows into the (m, x_dim) pre-clip originals.
    let mut orig_l = Vec::with_capacity(m * x_dim);
    let mut orig_u = Vec::with_capacity(m * x_dim);
    for &ci in with_lb {
        let s = ci * x_dim;
        orig_l.extend_from_slice(&lower_data[s..s + x_dim]);
        orig_u.extend_from_slice(&upper_data[s..s + x_dim]);
    }

    // Batched used-side coefficient-error discharge (I-A10): identical per-row
    // expressions to `fold_coeff_err_into_bias` — f64 penalty accumulated in
    // column order over the child's OWN box magnitudes, `(bias ∓ p)` widened
    // in f64 then directed-rounded, non-finite penalty → `∓inf` — landing in
    // a per-child bias matrix instead of a per-child `LinearBounds` clone.
    // The clip sign negation (upper direction) is applied AFTER the fold,
    // exactly as the reference folds first and negates at gather.
    let mut bias_used = vec![0f32; m * n_thr];
    let mut mag = vec![0f64; x_dim];
    for k in 0..m {
        let s = k * x_dim;
        for j in 0..x_dim {
            mag[j] = (orig_l[s + j] as f64)
                .abs()
                .max((orig_u[s + j] as f64).abs());
        }
        let slot = child_plane[k];
        let lb = plane_src[slot];
        let base_b = if verify_upper {
            lb.upper_b()
        } else {
            lb.lower_b()
        };
        let err = if verify_upper {
            lb.upper_a_err()
        } else {
            lb.lower_a_err()
        };
        let nrows = planes[slot].nrows;
        for i in 0..nrows.min(n_thr) {
            let mut bv = base_b[i];
            if let Some(err) = err {
                let mut p = 0.0f64;
                for j in 0..x_dim {
                    p += err[[i, j]] as f64 * mag[j];
                }
                if p != 0.0 {
                    if p.is_finite() {
                        bv = if verify_upper {
                            ny_tensor::next_up_f32((bv as f64 + p) as f32)
                        } else {
                            ny_tensor::next_down_f32((bv as f64 - p) as f32)
                        };
                    } else {
                        bv = if verify_upper {
                            f32::INFINITY
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                }
            }
            bias_used[k * n_thr + i] = if verify_upper { -bv } else { bv };
        }
    }

    // Single planes-based clip pass: m children x T threshold rows. Clause-aware
    // (#disj-cross-clause-clip-unsat): per-clause intersection + union bbox.
    let (clip_l, clip_u, verified_clip) = batched_relaxed_clip_from_planes(
        &orig_l,
        &orig_u,
        &planes,
        &child_plane,
        &bias_used,
        thresholds,
        clause_sizes,
        verify_upper,
        verifier.config.relaxed_clip_iterations,
        m,
        x_dim,
    )?;

    // Dispositions + pushes, identical order and identical trace points to the
    // reference body.
    let mut postclip: Vec<(f32, f32)> = Vec::with_capacity(n_thr);
    for (k, &ci) in with_lb.iter().enumerate() {
        let parent = &parents[parent_idx[ci]];
        if verified_clip[k] {
            lifecycle.domains_verified += 1;
            *domains_verified_by_clip += 1;
            trace!("batched clip: child {} verified by box infeasibility", k);
            continue;
        }

        let row_l = &clip_l[k * x_dim..(k + 1) * x_dim];
        let row_u = &clip_u[k * x_dim..(k + 1) * x_dim];
        concretize_postclip_lower_bounds_planes(
            row_l,
            row_u,
            &planes[child_plane[k]],
            &bias_used[k * n_thr..(k + 1) * n_thr],
            n_thr,
            &mut postclip,
        );
        if disjunctive_domain_verified(&postclip, thresholds, clause_sizes) {
            lifecycle.domains_verified += 1;
            *domains_verified_by_clip += 1;
            trace!(
                "batched clip: child {} verified by grouped disjunctive check",
                k
            );
            continue;
        }

        let row_l = ArrayD::from_shape_vec(IxDyn(&[x_dim]), row_l.to_vec()).map_err(|e| {
            NyError::InvalidSpec(format!("clip planes: clipped lower[{}]: {}", k, e))
        })?;
        let row_u = ArrayD::from_shape_vec(IxDyn(&[x_dim]), row_u.to_vec()).map_err(|e| {
            NyError::InvalidSpec(format!("clip planes: clipped upper[{}]: {}", k, e))
        })?;
        let child_input = build_child_input_owned(row_l, row_u, shape)?;
        queue.push(MultiObjInputDomain {
            input_bounds: Arc::new(child_input),
            obj_bounds: parent.obj_bounds.clone(),
            linear_bounds: None,
            depth: parent.child_depth,
            priority: parent.priority,
            needs_bounding: true,
            node_bounds_override: None,
            // Parent α slopes for the deferred-rebound warm overlay (step-2C/2D).
            inherited_alpha_state: parent.inherited_alpha_state.clone(),
        });
    }

    trace!("batched clip: processed {} children in single pass", m);
    #[cfg(test)]
    CLIP_PLANES_HANDLED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(true)
}

#[cfg(test)]
mod tests;
