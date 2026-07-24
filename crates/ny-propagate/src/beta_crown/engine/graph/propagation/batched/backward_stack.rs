// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain-stacked shared-dispatch backward for the batched dense-spec kernel
//! (#cgan-batched-stack).
//!
//! The batched CROWN backward previously looped domains one at a time through
//! `dispatch_backward_layer` for every non-Linear/non-ReLU node. For conv-heavy
//! graphs (cgan: 4 ConvTranspose2d + 4 Conv2d + BatchNorm per traversal) the
//! per-call fixed work (im2col materialization, weight reshape, engine
//! dispatch, and the sound f64 certified-error recompute) dominated: ~87% of a
//! 3.15 s 64-domain batch was ConvTranspose2d alone at 2 spec rows per call.
//!
//! This module stacks the active domains' `LinearBounds` rows into ONE
//! `(Σ rows_d) × n_inputs` backward call per whitelisted node and splits the
//! result rows back per domain.
//!
//! # Soundness (#vnncomp-aw-soundness discipline)
//!
//! - The whitelisted layers' CROWN backwards are **row-independent linear
//!   operators**: each output row of `dispatch_backward_layer` depends only on
//!   the corresponding input row (Conv/ConvTranspose contraction `A·W`,
//!   BatchNorm column scaling), including the certified per-coefficient error
//!   machinery (`num_objectives = rows`). Stacking rows and splitting the
//!   result is therefore exact row bookkeeping — identical math per row.
//! - The only box-dependent inputs consumed by these layers are
//!   **sound-widening** in the box: the pre-dispatch certified-error discharge
//!   `Σ_j max(|yl_j|,|yu_j|)·err_ij` (dispatch.rs) and BatchNorm's precompute
//!   error margin `w_err[i] = scale_err·max(|pre_l_i|,|pre_u_i|) + bias_err`
//!   (crown_scalar.rs) are monotone non-decreasing in the box magnitudes and
//!   are folded OUTWARD into the bias. The stacked call passes the elementwise
//!   **HULL** of the active domains' boxes, a superset of every per-domain
//!   box, so each row's widening is `>=` the per-domain path's widening:
//!   equal-or-looser, never tighter. Conv2d/ConvTranspose2d consume the
//!   pre-activation only for its SHAPE (identical across domains).
//! - **Fail-closed:** any shape mismatch, missing cache entry, non-`Single`
//!   dispatch result, row-count mismatch, or dispatch error makes the stacked
//!   attempt return `Ok(false)` and the caller runs the existing per-domain
//!   loop unchanged.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::{concatenate, s, Array2, Axis};
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

use crate::network::backward_dispatch::{
    dispatch_backward_layer, BackwardDispatchResult, DispatchContext,
};
use crate::{Layer, LinearBounds, MulBinaryRelaxationMode, NETWORK_INPUT};

use super::indexed_pending::IndexedPendingLinearBounds;

/// Layers whose batched backward may be domain-stacked into one dispatch call.
///
/// Whitelist rationale (see module docs): the backward must be a
/// row-independent linear operator whose only box consumption is a
/// sound-widening bias fold. Conv2d / ConvTranspose2d use the pre-activation
/// box for shape only; BatchNorm folds box-magnitude-monotone error margins
/// outward. Everything else stays on the per-domain loop.
pub(super) fn layer_supports_domain_stacking(layer: &Layer) -> bool {
    matches!(
        layer,
        Layer::Conv2d(_) | Layer::ConvTranspose2d(_) | Layer::BatchNorm(_)
    )
}

/// Elementwise hull (union box) of several `BoundedTensor`s.
///
/// Returns `None` on shape mismatch or invalid result (caller falls back).
fn hull_bounded_tensors(tensors: &[&BoundedTensor]) -> Option<BoundedTensor> {
    let first = tensors.first()?;
    if tensors.len() == 1 {
        return Some((*first).clone());
    }
    let shape = first.shape();
    let mut lower = first.lower().clone();
    let mut upper = first.upper().clone();
    for t in &tensors[1..] {
        if t.shape() != shape {
            return None;
        }
        for (dst, &src) in lower.iter_mut().zip(t.lower().iter()) {
            // NaN-safe: propagate NaN so BoundedTensor::new rejects it below
            // instead of silently keeping a possibly-wrong finite bound.
            if src.is_nan() || dst.is_nan() {
                *dst = f32::NAN;
            } else if src < *dst {
                *dst = src;
            }
        }
        for (dst, &src) in upper.iter_mut().zip(t.upper().iter()) {
            if src.is_nan() || dst.is_nan() {
                *dst = f32::NAN;
            } else if src > *dst {
                *dst = src;
            }
        }
    }
    BoundedTensor::new(lower, upper).ok()
}

/// Vertically stack the active domains' `LinearBounds` into one tall bound.
///
/// Error matrices: if ANY active domain carries a certified coefficient error,
/// every domain contributes its error rows (zeros when it carried `None`,
/// which is exactly the "exact coefficients" semantics of `None`).
fn stack_linear_bounds(active: &[(usize, &LinearBounds)]) -> Option<LinearBounds> {
    let ncols = active.first()?.1.num_inputs();
    if active.iter().any(|(_, lb)| lb.num_inputs() != ncols) {
        return None;
    }

    let lower_a_views: Vec<_> = active.iter().map(|(_, lb)| lb.lower_a().view()).collect();
    let upper_a_views: Vec<_> = active.iter().map(|(_, lb)| lb.upper_a().view()).collect();
    let lower_b_views: Vec<_> = active.iter().map(|(_, lb)| lb.lower_b().view()).collect();
    let upper_b_views: Vec<_> = active.iter().map(|(_, lb)| lb.upper_b().view()).collect();

    let lower_a = concatenate(Axis(0), &lower_a_views).ok()?;
    let upper_a = concatenate(Axis(0), &upper_a_views).ok()?;
    let lower_b = concatenate(Axis(0), &lower_b_views).ok()?;
    let upper_b = concatenate(Axis(0), &upper_b_views).ok()?;

    let any_err = active
        .iter()
        .any(|(_, lb)| lb.lower_a_err().is_some() || lb.upper_a_err().is_some());
    let (lower_a_err, upper_a_err) = if any_err {
        let zeros: Vec<Array2<f32>> = active
            .iter()
            .map(|(_, lb)| Array2::zeros((lb.num_outputs(), ncols)))
            .collect();
        let le_views: Vec<_> = active
            .iter()
            .zip(zeros.iter())
            .map(|((_, lb), z)| lb.lower_a_err().unwrap_or(z).view())
            .collect();
        let ue_views: Vec<_> = active
            .iter()
            .zip(zeros.iter())
            .map(|((_, lb), z)| lb.upper_a_err().unwrap_or(z).view())
            .collect();
        (
            Some(concatenate(Axis(0), &le_views).ok()?),
            Some(concatenate(Axis(0), &ue_views).ok()?),
        )
    } else {
        (None, None)
    };

    Some(LinearBounds {
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        lower_a_err,
        upper_a_err,
    })
}

/// Slice a stacked `Single` dispatch result back into per-domain bounds.
///
/// Row block `i` (of `active[i].1.num_outputs()` rows) belongs to domain
/// `active[i].0`. Values are copied bit-exactly — no re-validation, matching
/// what the per-domain loop would have accumulated.
fn split_stacked_result(
    stacked: &LinearBounds,
    active: &[(usize, &LinearBounds)],
) -> Option<Vec<(usize, LinearBounds)>> {
    let total_rows: usize = active.iter().map(|(_, lb)| lb.num_outputs()).sum();
    if stacked.num_outputs() != total_rows {
        return None;
    }
    let mut out = Vec::with_capacity(active.len());
    let mut offset = 0usize;
    for &(domain_idx, lb) in active {
        let rows = lb.num_outputs();
        let r = offset..offset + rows;
        let piece = LinearBounds {
            lower_a: stacked.lower_a().slice(s![r.clone(), ..]).to_owned(),
            lower_b: stacked.lower_b().slice(s![r.clone()]).to_owned(),
            upper_a: stacked.upper_a().slice(s![r.clone(), ..]).to_owned(),
            upper_b: stacked.upper_b().slice(s![r.clone()]).to_owned(),
            lower_a_err: stacked
                .lower_a_err()
                .map(|e| e.slice(s![r.clone(), ..]).to_owned()),
            upper_a_err: stacked
                .upper_a_err()
                .map(|e| e.slice(s![r.clone(), ..]).to_owned()),
        };
        out.push((domain_idx, piece));
        offset += rows;
    }
    Some(out)
}

/// Attempt one stacked dispatch across all active domains at this node.
///
/// Returns `Ok(true)` when the stacked call succeeded and all per-domain
/// results were accumulated into `node_linear_bounds`; `Ok(false)` when the
/// caller must fall back to the per-domain loop (nothing was accumulated).
#[allow(clippy::too_many_arguments)]
pub(super) fn try_stacked_dispatch(
    node_name: &str,
    node: &crate::GraphNode,
    node_lbs: &[Option<LinearBounds>],
    constrained_inputs: &[BoundedTensor],
    // #lsnc-shared-fwd: per-domain node-bounds caches are borrowed (a slice of
    // references), so the input-split lane can point every domain at ONE shared
    // warmup map with no per-domain deep clone. Read-only here.
    bounds_caches: &[&HashMap<String, Arc<BoundedTensor>>],
    node_linear_bounds: &mut IndexedPendingLinearBounds,
    engine: &dyn GemmEngine,
    deadline: Option<Instant>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
) -> Result<bool> {
    let active: Vec<(usize, &LinearBounds)> = node_lbs
        .iter()
        .enumerate()
        .filter_map(|(i, lb)| lb.as_ref().map(|lb| (i, lb)))
        .collect();
    // A single active domain gains nothing from stacking; keep the
    // per-domain path (bit-identical behavior).
    if active.len() < 2 {
        return Ok(false);
    }

    let stack_start = Instant::now();
    let Some(stacked_lb) = stack_linear_bounds(&active) else {
        tracing::debug!(
            node_name,
            "stacked batched backward: row-stack build failed, falling back"
        );
        return Ok(false);
    };

    // Hull boxes for every cache key the dispatch may consume: the node's own
    // output box (certified-error discharge) and its inputs (pre-activation).
    // A key missing from ANY active domain's cache is omitted from the hull
    // map — the discharge then degrades to conservative for all stacked rows,
    // which is looser than (a subset of) the per-domain path but sound.
    let mut hull_cache: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    let mut keys: Vec<&str> = Vec::with_capacity(1 + node.inputs.len());
    keys.push(node_name);
    for input in &node.inputs {
        if input != NETWORK_INPUT && input != node_name {
            keys.push(input.as_str());
        }
    }
    for key in keys {
        let entries: Vec<&BoundedTensor> = active
            .iter()
            .filter_map(|&(i, _)| bounds_caches[i].get(key).map(|a| a.as_ref()))
            .collect();
        if entries.len() != active.len() {
            continue;
        }
        match hull_bounded_tensors(&entries) {
            Some(hull) => {
                // Fresh Arc: hulled tensors are new allocations, never shared
                // ancestors (#cone-delta increment 2 aliasing rule).
                hull_cache.insert(key.to_string(), Arc::new(hull));
            }
            // Shape mismatch across domains: cannot build a box valid for all
            // stacked rows — fail closed to the per-domain loop.
            None => return Ok(false),
        }
    }

    let input_refs: Vec<&BoundedTensor> = active
        .iter()
        .map(|&(i, _)| &constrained_inputs[i])
        .collect();
    let Some(hull_input) = hull_bounded_tensors(&input_refs) else {
        return Ok(false);
    };

    // Resolve the (hulled) pre-activation for the first input.
    let pre_activation: &BoundedTensor = match node.inputs.first() {
        Some(name) if name == NETWORK_INPUT => &hull_input,
        Some(name) => match hull_cache.get(name.as_str()) {
            Some(b) => b.as_ref(),
            None => return Ok(false),
        },
        None => return Ok(false),
    };

    let ctx = DispatchContext {
        node_name,
        layer: &node.layer,
        inputs: &node.inputs,
        pre_activation,
        network_input: &hull_input,
        node_bounds: (&hull_cache).into(),
        engine: Some(engine),
        deadline,
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas,
        norm_inv_rms_override: None,
    };

    let dispatched = match dispatch_backward_layer(&ctx, &stacked_lb) {
        Ok(BackwardDispatchResult::Single(new_lb)) => *new_lb,
        Ok(_) => {
            // Whitelisted layers are unary Single-result ops; anything else is
            // unexpected — fall back rather than guess row ownership.
            tracing::debug!(
                node_name,
                layer = node.layer.layer_type(),
                "stacked batched backward: non-Single dispatch result, falling back to per-domain loop"
            );
            return Ok(false);
        }
        Err(err) => {
            tracing::debug!(
                node_name,
                layer = node.layer.layer_type(),
                %err,
                "stacked batched backward dispatch failed, falling back to per-domain loop"
            );
            return Ok(false);
        }
    };

    let Some(pieces) = split_stacked_result(&dispatched, &active) else {
        tracing::debug!(
            node_name,
            layer = node.layer.layer_type(),
            "stacked batched backward: row-count mismatch, falling back to per-domain loop"
        );
        return Ok(false);
    };

    let first_input = match node.inputs.first() {
        Some(name) => name.as_str(),
        None => return Ok(false),
    };
    for (domain_idx, piece) in pieces {
        node_linear_bounds.accumulate_name(first_input, piece, domain_idx)?;
    }
    tracing::debug!(
        node_name,
        layer = node.layer.layer_type(),
        n_active = active.len(),
        stacked_rows = stacked_lb.num_outputs(),
        elapsed_s = stack_start.elapsed().as_secs_f64(),
        "stacked batched backward dispatch fired"
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Array1};

    use super::*;

    fn mk_lb(scale: f32, rows: usize, cols: usize, with_err: bool) -> LinearBounds {
        let lower_a =
            Array2::from_shape_fn((rows, cols), |(i, j)| scale * (i as f32 + 1.0) + j as f32);
        let upper_a = &lower_a + 0.5;
        let lower_b = Array1::from_shape_fn(rows, |i| -(i as f32) - scale);
        let upper_b = Array1::from_shape_fn(rows, |i| i as f32 + scale);
        LinearBounds {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err: with_err.then(|| Array2::from_elem((rows, cols), 1e-6 * scale)),
            upper_a_err: with_err.then(|| Array2::from_elem((rows, cols), 2e-6 * scale)),
        }
    }

    #[test]
    fn stack_then_split_round_trips_rows_and_errs() {
        let a = mk_lb(1.0, 2, 3, false);
        let b = mk_lb(2.0, 2, 3, true);
        let active = vec![(0usize, &a), (3usize, &b)];

        let stacked = stack_linear_bounds(&active).expect("stack should succeed");
        assert_eq!(stacked.num_outputs(), 4);
        assert_eq!(stacked.num_inputs(), 3);
        // Domain without err contributes zero rows in the stacked err carrier.
        let le = stacked.lower_a_err().expect("stacked err present");
        assert!(le.slice(s![0..2, ..]).iter().all(|&v| v == 0.0));
        assert!(le.slice(s![2..4, ..]).iter().all(|&v| v == 1e-6 * 2.0));

        let pieces = split_stacked_result(&stacked, &active).expect("split should succeed");
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].0, 0);
        assert_eq!(pieces[1].0, 3);
        assert_eq!(pieces[0].1.lower_a(), a.lower_a());
        assert_eq!(pieces[0].1.upper_b(), a.upper_b());
        assert_eq!(pieces[1].1.upper_a(), b.upper_a());
        assert_eq!(
            pieces[1].1.lower_a_err().expect("err slice"),
            b.lower_a_err().expect("orig err")
        );
    }

    #[test]
    fn stack_rejects_column_mismatch() {
        let a = mk_lb(1.0, 2, 3, false);
        let b = mk_lb(1.0, 2, 4, false);
        assert!(stack_linear_bounds(&[(0, &a), (1, &b)]).is_none());
    }

    #[test]
    fn hull_is_containing_box_and_rejects_shape_mismatch() {
        let t1 = BoundedTensor::new(
            arr1(&[-1.0f32, 0.0]).into_dyn(),
            arr1(&[1.0f32, 2.0]).into_dyn(),
        )
        .unwrap();
        let t2 = BoundedTensor::new(
            arr1(&[-2.0f32, 0.5]).into_dyn(),
            arr1(&[0.5f32, 3.0]).into_dyn(),
        )
        .unwrap();
        let hull = hull_bounded_tensors(&[&t1, &t2]).expect("hull");
        assert_eq!(hull.lower().as_slice().unwrap(), &[-2.0, 0.0]);
        assert_eq!(hull.upper().as_slice().unwrap(), &[1.0, 3.0]);

        let t3 = BoundedTensor::new(
            arr2(&[[0.0f32, 0.0], [0.0, 0.0]]).into_dyn(),
            arr2(&[[1.0f32, 1.0], [1.0, 1.0]]).into_dyn(),
        )
        .unwrap();
        assert!(hull_bounded_tensors(&[&t1, &t3]).is_none());
    }
}
