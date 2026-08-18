// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Transactional host integration for sound GPU intermediate CROWN sweeps.
//!
//! The legacy production entry is deliberately a one-target vertical slice;
//! the comprehensive entry freezes the bounded all-target census. The canonical
//! plan builder accepts heterogeneous targets and emits the full
//! Unary/Identity/Add/Sub contract. A backend may decline an unsupported tape
//! only before dispatch. Every accepted result is validated and staged against
//! the current live map before any bound is published.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Instant;

use ndarray::{ArrayD, IxDyn};
use ny_core::{
    GpuBackwardOp, GpuBackwardSlot, GpuCrownBackward, GpuCrownLayer, GpuIntermediateInjection,
    GpuIntermediateSweepPlan, GpuIntermediateSweepRequest, NyError,
};
use ny_tensor::BoundedTensor;
use sha2::{Digest, Sha256};

use crate::layers::Layer;
use crate::network::{try_extract_single_gpu_layer, CrownDispatchPlan};
use crate::{GraphNetwork, NETWORK_INPUT};

use super::interm_refine::scoped_wide_demanded_sweep_targets_before;

const TRANSCRIPT_POLL_STRIDE: usize = 4_096;

/// Whether the legacy single-target route may be attempted after this call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::beta_crown::engine::graph) enum RootIntermediateSweepAttempt {
    /// A backend accepted a request and the complete validated transaction was
    /// processed. The count is the number of targets whose live box shrank.
    Completed(usize),
    /// No request was accepted and no GPU work was performed. A separately
    /// authorized legacy route may still be attempted within the same deadline.
    CleanDecline,
    /// An accepted request, validation, or publication transaction failed. Do
    /// not start a second verdict-bearing route from this state.
    Failed,
}

struct SelectedSweepTarget {
    node_name: String,
    selected_rows: Arc<[u32]>,
    frozen_bound: BoundedTensor,
    role: Option<RootIntermediateSweepTargetRole>,
}

struct FrozenSweepTarget {
    target_id: u64,
    node_name: String,
    target_shape: Arc<[usize]>,
    selected_rows: Arc<[u32]>,
    frozen_bound: BoundedTensor,
    role: Option<RootIntermediateSweepTargetRole>,
}

/// Role bound into the unified phase-resident target-set identity.
///
/// Legacy sweep routes use `None` and therefore retain their exact v1 target
/// transcript. A unified request uses one of these values for every target;
/// mixed role-bound/unbound requests are rejected during preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RootIntermediateSweepTargetRole {
    DenseMandatory,
    Comprehensive,
}

impl RootIntermediateSweepTargetRole {
    const fn identity_tag(self) -> u8 {
        match self {
            Self::DenseMandatory => 1,
            Self::Comprehensive => 2,
        }
    }
}

struct CanonicalSweepTape {
    ops_backward: Arc<[GpuBackwardOp]>,
    slot_dims: Arc<[usize]>,
    input_slot: GpuBackwardSlot,
    slot_names: Vec<String>,
}

struct PreparedRootIntermediateSweep {
    plan: GpuIntermediateSweepPlan,
    input_identity_sha256: [u8; 32],
    input_lower: Vec<f32>,
    input_upper: Vec<f32>,
    targets: Vec<FrozenSweepTarget>,
}

#[inline]
fn deadline_live(deadline: Instant) -> bool {
    Instant::now() < deadline
}

fn deadline_poll(deadline: Instant, context: &str) -> ny_core::Result<()> {
    if deadline_live(deadline) {
        Ok(())
    } else {
        Err(NyError::DeadlineExceeded(format!(
            "root intermediate sweep deadline expired {context}"
        )))
    }
}

fn digest_hex(digest: &[u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        // Writing into a pre-sized String is infallible. Keep this local
        // instead of pulling a formatting dependency into the proof crate.
        write!(&mut encoded, "{byte:02x}").expect("writing a SHA-256 digest into String");
    }
    encoded
}

fn finite_ordered_box(bounds: &BoundedTensor, deadline: Instant) -> bool {
    if bounds.lower().shape() != bounds.upper().shape() || bounds.lower().is_empty() {
        return false;
    }
    for (index, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper()).enumerate() {
        if index.is_multiple_of(TRANSCRIPT_POLL_STRIDE) && !deadline_live(deadline) {
            return false;
        }
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return false;
        }
    }
    deadline_live(deadline)
}

fn select_tightening_rows(
    bounds: &BoundedTensor,
    max_rows: usize,
    deadline: Instant,
) -> Option<Arc<[u32]>> {
    if max_rows == 0 || !finite_ordered_box(bounds, deadline) {
        return None;
    }
    let mut crossing = Vec::new();
    crossing.try_reserve_exact(bounds.len()).ok()?;
    let mut stable_nonpoint = Vec::new();
    stable_nonpoint.try_reserve_exact(bounds.len()).ok()?;
    for (index, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper()).enumerate() {
        if index.is_multiple_of(TRANSCRIPT_POLL_STRIDE) && !deadline_live(deadline) {
            return None;
        }
        if lower == upper {
            continue;
        }
        let candidate = (index, f64::from(upper) - f64::from(lower));
        if lower < 0.0 && upper > 0.0 {
            crossing.push(candidate);
        } else {
            stable_nonpoint.push(candidate);
        }
    }
    let order = |(a_index, a_width): &(usize, f64), (b_index, b_width): &(usize, f64)| {
        b_width
            .total_cmp(a_width)
            .then_with(|| a_index.cmp(b_index))
    };
    crossing.sort_by(order);
    stable_nonpoint.sort_by(order);
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(max_rows.min(crossing.len().saturating_add(stable_nonpoint.len())))
        .ok()?;
    for (index, _) in crossing.into_iter().chain(stable_nonpoint).take(max_rows) {
        selected.push(u32::try_from(index).ok()?);
    }
    selected.sort_unstable();
    (!selected.is_empty() && deadline_live(deadline)).then(|| Arc::from(selected))
}

/// Select a bounded per-target row set without letting one sign class starve
/// the other. When both crossing and sign-stable non-point rows exist, up to
/// four of each are retained first; remaining capacity is filled globally by
/// width. Canonical row IDs are sorted only after selection.
fn select_balanced_tightening_rows(
    bounds: &BoundedTensor,
    max_rows: usize,
    deadline: Instant,
) -> Option<Arc<[u32]>> {
    select_balanced_tightening_rows_chunk(bounds, 0, max_rows, deadline, None)
}

/// #interm-row-chunking: the same balanced ordering as
/// [`select_balanced_tightening_rows`], but returning the window
/// `[skip, skip + max_rows)` of it instead of always the head.
///
/// WHY: the comprehensive sweep is hard-capped at ~128 rows/target by device
/// memory (the backend declines 256 outright), which is ~2.8% coverage, and the
/// tightening benefit is LINEAR in coverage — 0.026 unstable-flips per row
/// measured at 1,152 rows, versus 0.021 for the full-coverage CPU pass. So the
/// only way to the coverage the root actually needs is to run the bounded sweep
/// REPEATEDLY over disjoint row windows and accumulate.
///
/// This is sound by construction and needs no new bound math: every sweep is
/// atomic over its own window, every commit is shrink-only intersect into the
/// frozen map, and windows are cut from ONE frozen transcript so they stay
/// disjoint and stable even as earlier chunks tighten the live bounds. A chunk
/// that returns `None` (window past the end, deadline, allocation refusal) simply
/// ends the accumulation with everything already committed still valid.
///
/// #root-objective-directed-rows: `influence`, when `Some`, is a per-neuron
/// magnitude of this target's effect on the OBJECTIVE ensemble (see
/// `objective_row_influence`). The ordering key becomes `influence * width`
/// instead of `width` alone, so the bounded row budget is spent on the neurons
/// that actually move the margin rather than on whichever intervals happen to be
/// widest. Width alone is objective-BLIND: a very wide neuron the output barely
/// reads is ranked above a moderate one the margin depends on directly.
///
/// Advisory-only. The key selects WHICH rows to tighten; every selected row is
/// then bounded by exactly the same sound backend sweep, and the commit stays a
/// shrink-only intersect. A wrong or stale influence vector can only waste rows,
/// never admit an unsound bound. `None` restores the width ordering byte for byte.
fn select_balanced_tightening_rows_chunk(
    bounds: &BoundedTensor,
    skip: usize,
    max_rows: usize,
    deadline: Instant,
    influence: Option<&[f32]>,
) -> Option<Arc<[u32]>> {
    if max_rows == 0 || !finite_ordered_box(bounds, deadline) {
        return None;
    }
    // The window is cut AFTER the balanced ordering is built, so `max_rows` here
    // bounds the returned window while the ordering itself still spans the whole
    // candidate set.
    let max_rows = skip.checked_add(max_rows)?;
    let mut crossing = Vec::new();
    crossing.try_reserve_exact(bounds.len()).ok()?;
    let mut stable = Vec::new();
    stable.try_reserve_exact(bounds.len()).ok()?;
    for (index, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper()).enumerate() {
        if index.is_multiple_of(TRANSCRIPT_POLL_STRIDE) && !deadline_live(deadline) {
            return None;
        }
        if lower == upper {
            continue;
        }
        // #root-objective-directed-rows: rank by objective influence x width when
        // an influence vector is supplied, else by width alone (legacy).
        let width = f64::from(upper) - f64::from(lower);
        let key = match influence {
            Some(weights) => f64::from(weights.get(index).copied().unwrap_or(0.0)) * width,
            None => width,
        };
        let candidate = (index, key);
        if lower < 0.0 && upper > 0.0 {
            crossing.push(candidate);
        } else {
            stable.push(candidate);
        }
    }
    if crossing.is_empty() && stable.is_empty() {
        return None;
    }
    let order = |(a_index, a_width): &(usize, f64), (b_index, b_width): &(usize, f64)| {
        b_width
            .total_cmp(a_width)
            .then_with(|| a_index.cmp(b_index))
    };
    crossing.sort_by(order);
    stable.sort_by(order);

    let total = crossing.len().checked_add(stable.len())?;
    let selected_capacity = max_rows.min(total);
    let mut chosen = Vec::new();
    chosen.try_reserve_exact(selected_capacity).ok()?;
    let balanced_each = if crossing.is_empty() || stable.is_empty() {
        0
    } else {
        4.min(selected_capacity / 2)
    };
    let crossing_floor = balanced_each.min(crossing.len());
    let stable_floor = balanced_each.min(stable.len());
    chosen.extend(crossing.iter().take(crossing_floor).copied());
    chosen.extend(stable.iter().take(stable_floor).copied());

    let mut remaining = Vec::new();
    remaining
        .try_reserve_exact(total.saturating_sub(chosen.len()))
        .ok()?;
    remaining.extend(crossing.into_iter().skip(crossing_floor));
    remaining.extend(stable.into_iter().skip(stable_floor));
    remaining.sort_by(order);
    chosen.extend(
        remaining
            .into_iter()
            .take(selected_capacity.saturating_sub(chosen.len())),
    );
    // #interm-row-chunking: `chosen` is the balanced ordering truncated to
    // `skip + window`; drop the first `skip` so successive chunks are disjoint.
    // A `skip` past the end yields an empty window and ends the accumulation.
    if skip >= chosen.len() {
        return None;
    }
    let chosen = &chosen[skip..];
    let mut selected = Vec::new();
    selected.try_reserve_exact(chosen.len()).ok()?;
    for (index, _) in chosen {
        selected.push(u32::try_from(*index).ok()?);
    }
    selected.sort_unstable();
    (!selected.is_empty() && deadline_live(deadline)).then(|| Arc::from(selected))
}

fn resolve_bounds<'a>(
    name: &str,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
) -> Option<&'a BoundedTensor> {
    if name == NETWORK_INPUT {
        Some(input)
    } else {
        bounds.get(name)
    }
}

fn live_ancestor_mask(
    graph: &GraphNetwork,
    dispatch: &CrownDispatchPlan,
    target_indices: &[usize],
    deadline: Instant,
) -> Option<Vec<bool>> {
    let mut live = vec![false; dispatch.node_count() + 1];
    let mut pending = Vec::new();
    pending.try_reserve_exact(dispatch.node_count() + 1).ok()?;
    pending.extend_from_slice(target_indices);
    let mut visited = 0usize;
    while let Some(index) = pending.pop() {
        if visited.is_multiple_of(64) && !deadline_live(deadline) {
            return None;
        }
        visited = visited.checked_add(1)?;
        if *live.get(index)? {
            continue;
        }
        live[index] = true;
        if dispatch.is_network_input(index) {
            continue;
        }
        let node = graph.nodes.get(dispatch.name_of(index))?;
        for input_name in node.inputs() {
            pending.push(dispatch.index_of(input_name)?);
        }
    }
    live[dispatch.network_input_idx] = true;
    Some(live)
}

fn build_canonical_tape(
    graph: &GraphNetwork,
    dispatch: &CrownDispatchPlan,
    target_indices: &[usize],
    bounds: &HashMap<String, BoundedTensor>,
    input: &BoundedTensor,
    deadline: Instant,
) -> Option<(CanonicalSweepTape, Vec<Option<GpuBackwardSlot>>)> {
    let live = live_ancestor_mask(graph, dispatch, target_indices, deadline)?;
    let mut slots_by_graph_index = vec![None; dispatch.node_count() + 1];
    let mut slot_names = Vec::new();
    slot_names
        .try_reserve_exact(live.iter().filter(|&&is_live| is_live).count())
        .ok()?;

    for &graph_index in &dispatch.reverse_order {
        if !live[graph_index] {
            continue;
        }
        let slot = GpuBackwardSlot(u32::try_from(slot_names.len()).ok()?);
        slots_by_graph_index[graph_index] = Some(slot);
        slot_names.push(dispatch.name_of(graph_index).to_string());
    }
    let input_slot = GpuBackwardSlot(u32::try_from(slot_names.len()).ok()?);
    slots_by_graph_index[dispatch.network_input_idx] = Some(input_slot);
    slot_names.push(NETWORK_INPUT.to_string());

    let mut slot_dims = Vec::new();
    slot_dims.try_reserve_exact(slot_names.len()).ok()?;
    for (slot_index, name) in slot_names.iter().enumerate() {
        if slot_index.is_multiple_of(64) && !deadline_live(deadline) {
            return None;
        }
        let bound = resolve_bounds(name, input, bounds)?;
        if !finite_ordered_box(bound, deadline) {
            return None;
        }
        slot_dims.push(bound.len());
    }

    let mut ops = Vec::new();
    ops.try_reserve_exact(slot_names.len().saturating_sub(1))
        .ok()?;
    for (op_index, &graph_index) in dispatch.reverse_order.iter().enumerate() {
        if !live[graph_index] {
            continue;
        }
        if op_index.is_multiple_of(64) && !deadline_live(deadline) {
            return None;
        }
        let name = dispatch.name_of(graph_index);
        let node = graph.nodes.get(name)?;
        let output = slots_by_graph_index[graph_index]?;
        match node.layer() {
            Layer::Add(_) | Layer::Sub(_) => {
                if node.inputs().len() != 2 {
                    return None;
                }
                let lhs_name = &node.inputs()[0];
                let rhs_name = &node.inputs()[1];
                let lhs_index = dispatch.index_of(lhs_name)?;
                let rhs_index = dispatch.index_of(rhs_name)?;
                let lhs = slots_by_graph_index[lhs_index]?;
                let rhs = slots_by_graph_index[rhs_index]?;
                let output_bounds = resolve_bounds(name, input, bounds)?;
                let lhs_bounds = resolve_bounds(lhs_name, input, bounds)?;
                let rhs_bounds = resolve_bounds(rhs_name, input, bounds)?;
                if output_bounds.shape() != lhs_bounds.shape()
                    || output_bounds.shape() != rhs_bounds.shape()
                {
                    return None;
                }
                ops.push(match node.layer() {
                    Layer::Add(_) => GpuBackwardOp::Add { output, lhs, rhs },
                    Layer::Sub(_) => GpuBackwardOp::Sub { output, lhs, rhs },
                    _ => unreachable!("outer match restricts Add/Sub"),
                });
            }
            Layer::Flatten(_) | Layer::Reshape(_) => {
                if node.inputs().len() != 1 {
                    return None;
                }
                let input_index = dispatch.index_of(&node.inputs()[0])?;
                let input_slot = slots_by_graph_index[input_index]?;
                ops.push(GpuBackwardOp::Identity {
                    output,
                    input: input_slot,
                });
            }
            _ => {
                if node.inputs().len() != 1 {
                    return None;
                }
                let input_name = &node.inputs()[0];
                let input_index = dispatch.index_of(input_name)?;
                let input_slot = slots_by_graph_index[input_index]?;
                let pre_activation = resolve_bounds(input_name, input, bounds)?;
                let mut layers = Vec::with_capacity(1);
                try_extract_single_gpu_layer(node.layer(), pre_activation, &mut layers)?;
                if layers.len() != 1 {
                    return None;
                }
                ops.push(GpuBackwardOp::Unary {
                    output,
                    input: input_slot,
                    layer: Box::new(layers.pop()?),
                });
            }
        }
    }

    Some((
        CanonicalSweepTape {
            ops_backward: Arc::from(ops),
            slot_dims: Arc::from(slot_dims),
            input_slot,
            slot_names,
        },
        slots_by_graph_index,
    ))
}

fn hash_u64(hasher: &mut Sha256, value: usize) -> Option<()> {
    hasher.update(u64::try_from(value).ok()?.to_le_bytes());
    Some(())
}

fn hash_bytes(hasher: &mut Sha256, value: &[u8]) -> Option<()> {
    hash_u64(hasher, value.len())?;
    hasher.update(value);
    Some(())
}

fn hash_shape(hasher: &mut Sha256, shape: &[usize]) -> Option<()> {
    hash_u64(hasher, shape.len())?;
    for &dimension in shape {
        hash_u64(hasher, dimension)?;
    }
    Some(())
}

fn hash_f32_slice(hasher: &mut Sha256, values: &[f32], deadline: Instant) -> Option<()> {
    hash_u64(hasher, values.len())?;
    for (index, value) in values.iter().enumerate() {
        if index.is_multiple_of(TRANSCRIPT_POLL_STRIDE) && !deadline_live(deadline) {
            return None;
        }
        hasher.update(value.to_bits().to_le_bytes());
    }
    Some(())
}

fn hash_u32_slice(hasher: &mut Sha256, values: &[u32], deadline: Instant) -> Option<()> {
    hash_u64(hasher, values.len())?;
    for (index, value) in values.iter().enumerate() {
        if index.is_multiple_of(TRANSCRIPT_POLL_STRIDE) && !deadline_live(deadline) {
            return None;
        }
        hasher.update(value.to_le_bytes());
    }
    Some(())
}

fn hash_optional_f32_arc(
    hasher: &mut Sha256,
    values: Option<&Arc<[f32]>>,
    deadline: Instant,
) -> Option<()> {
    match values {
        Some(values) => {
            hasher.update([1]);
            hash_f32_slice(hasher, values, deadline)
        }
        None => {
            hasher.update([0]);
            Some(())
        }
    }
}

fn hash_gpu_layer(hasher: &mut Sha256, layer: &GpuCrownLayer, deadline: Instant) -> Option<()> {
    match layer {
        GpuCrownLayer::Linear {
            weight,
            bias,
            out_features,
            in_features,
            cert_err,
        } => {
            hasher.update([0]);
            hash_u64(hasher, *out_features)?;
            hash_u64(hasher, *in_features)?;
            hasher.update(cert_err.weight_rel_err.to_bits().to_le_bytes());
            hasher.update(cert_err.bias_abs_err.to_bits().to_le_bytes());
            hash_f32_slice(hasher, weight, deadline)?;
            hash_optional_f32_arc(hasher, bias.as_ref(), deadline)
        }
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            num_neurons,
        } => {
            hasher.update([1]);
            hash_u64(hasher, *num_neurons)?;
            hash_f32_slice(hasher, lower_slope, deadline)?;
            hash_f32_slice(hasher, upper_slope, deadline)?;
            hash_f32_slice(hasher, lower_intercept, deadline)?;
            hash_f32_slice(hasher, upper_intercept, deadline)
        }
        GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            in_h,
            in_w,
            cert_err,
        } => {
            hasher.update([2]);
            for value in [
                *out_channels,
                *in_channels,
                *kernel_h,
                *kernel_w,
                *stride_h,
                *stride_w,
                *pad_h,
                *pad_w,
                *out_h,
                *out_w,
                *in_h,
                *in_w,
            ] {
                hash_u64(hasher, value)?;
            }
            hasher.update(cert_err.weight_rel_err.to_bits().to_le_bytes());
            hasher.update(cert_err.bias_abs_err.to_bits().to_le_bytes());
            hash_f32_slice(hasher, weight_col, deadline)?;
            hash_optional_f32_arc(hasher, bias_expanded.as_ref(), deadline)
        }
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope,
            cross_slope,
            upper_neg_slope,
            cross_intercept,
            num_neurons,
        } => {
            hasher.update([3]);
            hash_u64(hasher, *num_neurons)?;
            hash_f32_slice(hasher, lower_pos_slope, deadline)?;
            hash_f32_slice(hasher, cross_slope, deadline)?;
            hash_f32_slice(hasher, upper_neg_slope, deadline)?;
            hash_f32_slice(hasher, cross_intercept, deadline)
        }
        GpuCrownLayer::MaxPool2d {
            routing,
            ibp_lower,
            ibp_upper,
            input_dim,
            output_dim,
        } => {
            hasher.update([4]);
            hash_u64(hasher, *input_dim)?;
            hash_u64(hasher, *output_dim)?;
            hash_u32_slice(hasher, routing, deadline)?;
            hash_f32_slice(hasher, ibp_lower, deadline)?;
            hash_f32_slice(hasher, ibp_upper, deadline)
        }
    }
}

fn graph_identity(tape: &CanonicalSweepTape, deadline: Instant) -> Option<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"ny.gpu-intermediate-sweep.graph.v1\0");
    hash_u64(&mut hasher, tape.slot_names.len())?;
    for (index, (name, &dimension)) in tape
        .slot_names
        .iter()
        .zip(tape.slot_dims.iter())
        .enumerate()
    {
        if index.is_multiple_of(64) && !deadline_live(deadline) {
            return None;
        }
        hash_bytes(&mut hasher, name.as_bytes())?;
        hash_u64(&mut hasher, dimension)?;
    }
    hash_u64(&mut hasher, tape.ops_backward.len())?;
    for (index, op) in tape.ops_backward.iter().enumerate() {
        if index.is_multiple_of(64) && !deadline_live(deadline) {
            return None;
        }
        match op {
            GpuBackwardOp::Unary {
                output,
                input,
                layer,
            } => {
                hasher.update([0]);
                hasher.update(output.0.to_le_bytes());
                hasher.update(input.0.to_le_bytes());
                hash_gpu_layer(&mut hasher, layer, deadline)?;
            }
            GpuBackwardOp::Identity { output, input } => {
                hasher.update([1]);
                hasher.update(output.0.to_le_bytes());
                hasher.update(input.0.to_le_bytes());
            }
            GpuBackwardOp::Add { output, lhs, rhs } => {
                hasher.update([2]);
                hasher.update(output.0.to_le_bytes());
                hasher.update(lhs.0.to_le_bytes());
                hasher.update(rhs.0.to_le_bytes());
            }
            GpuBackwardOp::Sub { output, lhs, rhs } => {
                hasher.update([3]);
                hasher.update(output.0.to_le_bytes());
                hasher.update(lhs.0.to_le_bytes());
                hasher.update(rhs.0.to_le_bytes());
            }
        }
    }
    deadline_live(deadline).then(|| hasher.finalize().into())
}

fn hash_box(hasher: &mut Sha256, bounds: &BoundedTensor, deadline: Instant) -> Option<()> {
    hash_shape(hasher, bounds.shape())?;
    let lower = bounds.lower().as_slice()?;
    let upper = bounds.upper().as_slice()?;
    hash_f32_slice(hasher, lower, deadline)?;
    hash_f32_slice(hasher, upper, deadline)
}

fn bounds_identity(
    tape: &CanonicalSweepTape,
    bounds: &HashMap<String, BoundedTensor>,
    deadline: Instant,
) -> Option<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"ny.gpu-intermediate-sweep.bounds-box-alpha-none-beta-none.v1\0");
    hash_u64(&mut hasher, tape.slot_names.len().checked_sub(1)?)?;
    for (index, name) in tape
        .slot_names
        .iter()
        .take(tape.slot_names.len().saturating_sub(1))
        .enumerate()
    {
        if index.is_multiple_of(64) && !deadline_live(deadline) {
            return None;
        }
        hash_bytes(&mut hasher, name.as_bytes())?;
        hash_box(&mut hasher, bounds.get(name)?, deadline)?;
    }
    deadline_live(deadline).then(|| hasher.finalize().into())
}

fn input_identity(input: &BoundedTensor, deadline: Instant) -> Option<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"ny.gpu-intermediate-sweep.input-box-f32.v1\0");
    hash_box(&mut hasher, input, deadline)?;
    deadline_live(deadline).then(|| hasher.finalize().into())
}

fn target_set_identity(
    targets: &[FrozenSweepTarget],
    injections: &[GpuIntermediateInjection],
    deadline: Instant,
) -> Option<[u8; 32]> {
    if targets.len() != injections.len() {
        return None;
    }
    let role_bound = targets.first()?.role.is_some();
    if targets
        .iter()
        .any(|target| target.role.is_some() != role_bound)
    {
        return None;
    }
    let mut hasher = Sha256::new();
    if role_bound {
        hasher.update(b"ny.gpu-intermediate-sweep.target-set.phase-resident.v2\0");
    } else {
        // Preserve the exact legacy identity domain when the new lever is off.
        hasher.update(b"ny.gpu-intermediate-sweep.target-set.wide-demanded.v1\0");
    }
    hash_u64(&mut hasher, targets.len())?;
    for (index, (target, injection)) in targets.iter().zip(injections).enumerate() {
        if index.is_multiple_of(64) && !deadline_live(deadline) {
            return None;
        }
        if let Some(role) = target.role {
            hasher.update([role.identity_tag()]);
        }
        hash_bytes(&mut hasher, target.node_name.as_bytes())?;
        hasher.update(target.target_id.to_le_bytes());
        hasher.update(injection.slot.0.to_le_bytes());
        hash_shape(&mut hasher, &target.target_shape)?;
        hash_u32_slice(&mut hasher, &target.selected_rows, deadline)?;
        hash_u64(&mut hasher, injection.row_offset)?;
    }
    deadline_live(deadline).then(|| hasher.finalize().into())
}

fn prepare_root_intermediate_sweep(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    selected: Vec<SelectedSweepTarget>,
    deadline: Instant,
) -> Option<PreparedRootIntermediateSweep> {
    if selected.is_empty() || !finite_ordered_box(input, deadline) {
        return None;
    }
    let dispatch = graph.dispatch_plan().ok()?;
    if !deadline_live(deadline) {
        return None;
    }
    let mut seen = HashSet::new();
    seen.try_reserve(selected.len()).ok()?;
    let mut target_indices = Vec::new();
    target_indices.try_reserve_exact(selected.len()).ok()?;
    for target in &selected {
        if !seen.insert(target.node_name.as_str()) {
            return None;
        }
        let graph_index = dispatch.index_of(&target.node_name)?;
        if dispatch.is_network_input(graph_index) {
            return None;
        }
        target_indices.push(graph_index);
    }

    let (tape, slots_by_graph_index) =
        build_canonical_tape(graph, dispatch, &target_indices, bounds, input, deadline)?;
    let mut targets = Vec::new();
    targets.try_reserve_exact(selected.len()).ok()?;
    for (selected, graph_index) in selected.into_iter().zip(target_indices) {
        let slot = slots_by_graph_index[graph_index]?;
        let target_id = u64::try_from(graph_index).ok()?;
        let target_shape: Arc<[usize]> = Arc::from(selected.frozen_bound.shape().to_vec());
        targets.push((
            slot,
            FrozenSweepTarget {
                target_id,
                node_name: selected.node_name,
                target_shape,
                selected_rows: selected.selected_rows,
                frozen_bound: selected.frozen_bound,
                role: selected.role,
            },
        ));
    }
    targets.sort_by_key(|(slot, target)| (*slot, target.target_id));

    let mut injections = Vec::new();
    injections.try_reserve_exact(targets.len()).ok()?;
    let mut frozen_targets = Vec::new();
    frozen_targets.try_reserve_exact(targets.len()).ok()?;
    let mut row_offset = 0usize;
    for (slot, target) in targets {
        injections.push(GpuIntermediateInjection {
            target_id: target.target_id,
            slot,
            target_shape: Arc::clone(&target.target_shape),
            selected_rows: Arc::clone(&target.selected_rows),
            row_offset,
        });
        row_offset = row_offset.checked_add(target.selected_rows.len())?;
        frozen_targets.push(target);
    }

    let graph_identity_sha256 = graph_identity(&tape, deadline)?;
    let bounds_identity_sha256 = bounds_identity(&tape, bounds, deadline)?;
    let target_set_identity_sha256 = target_set_identity(&frozen_targets, &injections, deadline)?;
    let input_identity_sha256 = input_identity(input, deadline)?;
    let plan = GpuIntermediateSweepPlan {
        graph_identity_sha256,
        bounds_identity_sha256,
        target_set_identity_sha256,
        ops_backward: tape.ops_backward,
        slot_dims: tape.slot_dims,
        input_slot: tape.input_slot,
        injections: Arc::from(injections),
        total_rows: row_offset,
    };
    plan.validate().ok()?;
    if !deadline_live(deadline) {
        return None;
    }
    Some(PreparedRootIntermediateSweep {
        plan,
        input_identity_sha256,
        input_lower: input.lower().iter().copied().collect(),
        input_upper: input.upper().iter().copied().collect(),
        targets: frozen_targets,
    })
}

fn candidate_from_result(
    target: &FrozenSweepTarget,
    result: &ny_core::GpuIntermediateTargetResult,
) -> Option<BoundedTensor> {
    if result.target_id != target.target_id
        || result.selected_rows != target.selected_rows
        || result.lower_bounds.len() != target.selected_rows.len()
        || result.upper_bounds.len() != target.selected_rows.len()
    {
        return None;
    }
    let mut lower: Vec<f32> = target.frozen_bound.lower().iter().copied().collect();
    let mut upper: Vec<f32> = target.frozen_bound.upper().iter().copied().collect();
    for ((&row, &new_lower), &new_upper) in target
        .selected_rows
        .iter()
        .zip(&result.lower_bounds)
        .zip(&result.upper_bounds)
    {
        let row = row as usize;
        *lower.get_mut(row)? = new_lower;
        *upper.get_mut(row)? = new_upper;
    }
    let shape = IxDyn(&target.target_shape);
    let lower = ArrayD::from_shape_vec(shape.clone(), lower).ok()?;
    let upper = ArrayD::from_shape_vec(shape, upper).ok()?;
    BoundedTensor::new(lower, upper).ok()
}

fn publish_validated_batch(
    bounds: &mut HashMap<String, BoundedTensor>,
    targets: &[FrozenSweepTarget],
    results: &[ny_core::GpuIntermediateTargetResult],
    deadline: Instant,
    authority_live: impl FnOnce() -> bool,
) -> Option<usize> {
    if targets.len() != results.len() || !deadline_live(deadline) {
        return None;
    }
    let mut staged: Vec<Option<BoundedTensor>> = Vec::new();
    staged.try_reserve_exact(targets.len()).ok()?;
    let mut tightened = 0usize;
    for (index, (target, result)) in targets.iter().zip(results).enumerate() {
        if index.is_multiple_of(64) && !deadline_live(deadline) {
            return None;
        }
        let candidate = candidate_from_result(target, result)?;
        let current = bounds.get(&target.node_name)?;
        if current.shape() != target.target_shape.as_ref() {
            return None;
        }
        let (mut intersection, disjoint) = current
            .intersection_per_element_with_poll(&candidate, || {
                deadline_poll(deadline, "while staging live-map intersections")
            })
            .ok()??;
        if disjoint != 0 {
            return None;
        }
        let strictly_tighter = intersection
            .lower()
            .iter()
            .zip(intersection.upper())
            .zip(current.lower().iter().zip(current.upper()))
            .any(|((&new_lower, &new_upper), (&old_lower, &old_upper))| {
                new_lower > old_lower || new_upper < old_upper
            });
        if strictly_tighter {
            // Intersecting endpoint boxes does not invalidate an independently
            // proven ball that already encloses the same live tensor values.
            // Preserve it exactly so a downstream normalization→Linear
            // consumer cannot silently lose tightening metadata.
            if let Some(l2) = current.l2_constraint().cloned() {
                intersection = intersection.with_l2_constraint(l2);
                if !intersection.has_l2_constraint() {
                    return None;
                }
            }
            tightened = tightened.checked_add(1)?;
            staged.push(Some(intersection));
        } else {
            // Preserve the current tensor, including any L2 annotation, rather
            // than replacing it with an equal elementwise intersection.
            staged.push(None);
        }
    }
    if !targets.iter().all(|target| {
        bounds
            .get(&target.node_name)
            .is_some_and(|current| current.shape() == target.target_shape.as_ref())
    }) {
        return None;
    }

    // Sample the exact retained authority after every fallible scratch/live-map
    // check and immediately adjacent to the first assignment. Authority revoked
    // during O(target_dim) staging can never publish a stale result.
    if !deadline_live(deadline) || !authority_live() {
        return None;
    }

    // Everything fallible is complete. Assignment into existing entries cannot
    // allocate, so no caller-visible prefix can escape this loop.
    for (target, replacement) in targets.iter().zip(staged) {
        if let Some(replacement) = replacement {
            *bounds
                .get_mut(&target.node_name)
                .expect("targets were revalidated immediately before atomic commit") = replacement;
        }
    }
    Some(tightened)
}

fn frozen_bound_matches(left: &BoundedTensor, right: &BoundedTensor) -> bool {
    left.shape() == right.shape()
        && left
            .lower()
            .iter()
            .zip(right.lower())
            .all(|(&a, &b)| a.to_bits() == b.to_bits())
        && left
            .upper()
            .iter()
            .zip(right.upper())
            .all(|(&a, &b)| a.to_bits() == b.to_bits())
        && l2_constraint_bits_match(left.l2_constraint(), right.l2_constraint())
}

fn l2_constraint_bits_match(
    left: Option<&ny_tensor::L2Constraint>,
    right: Option<&ny_tensor::L2Constraint>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.axis() == right.axis()
                && left.center().shape() == right.center().shape()
                && left
                    .center()
                    .iter()
                    .zip(right.center())
                    .all(|(&a, &b)| a.to_bits() == b.to_bits())
                && left.radius().shape() == right.radius().shape()
                && left
                    .radius()
                    .iter()
                    .zip(right.radius())
                    .all(|(&a, &b)| a.to_bits() == b.to_bits())
        }
        (None, Some(_)) | (Some(_), None) => false,
    }
}

/// Publish one role-bound unified transaction.
///
/// Unlike the legacy publisher, this requires every live target to remain
/// bit-identical to the frozen snapshot. Selected rows retain the old endpoint
/// bits on equality and take only strict `max(lower)` / `min(upper)` progress;
/// unselected rows are copied verbatim. Every candidate is staged before the
/// final authority sample, so a bad or late target cannot publish a prefix.
fn publish_validated_role_bound_batch(
    bounds: &mut HashMap<String, BoundedTensor>,
    targets: &[FrozenSweepTarget],
    results: &[ny_core::GpuIntermediateTargetResult],
    deadline: Instant,
    authority_live: impl FnOnce() -> bool,
) -> Option<usize> {
    if targets.len() != results.len()
        || targets.iter().any(|target| target.role.is_none())
        || !deadline_live(deadline)
    {
        return None;
    }
    let mut staged: Vec<Option<BoundedTensor>> = Vec::new();
    staged.try_reserve_exact(targets.len()).ok()?;
    let mut tightened_targets = 0usize;
    for (target_index, (target, result)) in targets.iter().zip(results).enumerate() {
        if target_index.is_multiple_of(64) && !deadline_live(deadline) {
            return None;
        }
        if result.target_id != target.target_id
            || result.selected_rows != target.selected_rows
            || result.lower_bounds.len() != target.selected_rows.len()
            || result.upper_bounds.len() != target.selected_rows.len()
        {
            return None;
        }
        let current = bounds.get(&target.node_name)?;
        if !frozen_bound_matches(current, &target.frozen_bound) {
            return None;
        }
        let mut lower: Vec<f32> = current.lower().iter().copied().collect();
        let mut upper: Vec<f32> = current.upper().iter().copied().collect();
        let mut target_changed = false;
        for (row_index, ((&row, &new_lower), &new_upper)) in target
            .selected_rows
            .iter()
            .zip(&result.lower_bounds)
            .zip(&result.upper_bounds)
            .enumerate()
        {
            if row_index.is_multiple_of(TRANSCRIPT_POLL_STRIDE) && !deadline_live(deadline) {
                return None;
            }
            if !new_lower.is_finite() || !new_upper.is_finite() || new_lower > new_upper {
                return None;
            }
            let row = row as usize;
            let old_lower = *lower.get(row)?;
            let old_upper = *upper.get(row)?;
            let best_lower = if new_lower > old_lower {
                new_lower
            } else {
                old_lower
            };
            let best_upper = if new_upper < old_upper {
                new_upper
            } else {
                old_upper
            };
            if best_lower > best_upper {
                return None;
            }
            target_changed |= best_lower > old_lower || best_upper < old_upper;
            lower[row] = best_lower;
            upper[row] = best_upper;
        }
        if target_changed {
            let shape = IxDyn(&target.target_shape);
            let lower = ArrayD::from_shape_vec(shape.clone(), lower).ok()?;
            let upper = ArrayD::from_shape_vec(shape, upper).ok()?;
            let mut replacement = BoundedTensor::new(lower, upper).ok()?;
            if let Some(l2) = current.l2_constraint().cloned() {
                replacement = replacement.with_l2_constraint(l2);
                if !replacement.has_l2_constraint() {
                    return None;
                }
            }
            tightened_targets = tightened_targets.checked_add(1)?;
            staged.push(Some(replacement));
        } else {
            staged.push(None);
        }
    }

    // Catch a late live-map change after staging, then sample the retained
    // typed authority immediately adjacent to the allocation-free commit.
    if !targets.iter().all(|target| {
        bounds
            .get(&target.node_name)
            .is_some_and(|current| frozen_bound_matches(current, &target.frozen_bound))
    }) || !deadline_live(deadline)
        || !authority_live()
    {
        return None;
    }
    for (target, replacement) in targets.iter().zip(staged) {
        if let Some(replacement) = replacement {
            *bounds
                .get_mut(&target.node_name)
                .expect("role-bound targets were revalidated before atomic commit") = replacement;
        }
    }
    Some(tightened_targets)
}

fn row_retry_ladder(max_rows: usize) -> Vec<usize> {
    if max_rows == 0 {
        return Vec::new();
    }
    let floor = max_rows.min(32);
    let mut rows = max_rows;
    let mut ladder = Vec::with_capacity(5);
    loop {
        ladder.push(rows);
        if rows == floor {
            break;
        }
        rows = (rows / 2).max(floor);
    }
    ladder
}

fn comprehensive_row_retry_ladder(preferred_rows: usize, minimum_rows: usize) -> Vec<usize> {
    if minimum_rows == 0 || preferred_rows < minimum_rows {
        return Vec::new();
    }
    let mut ladder = Vec::with_capacity(4);
    let mut rows = preferred_rows;
    loop {
        ladder.push(rows);
        if rows == minimum_rows {
            break;
        }
        rows = (rows / 2).max(minimum_rows);
    }
    ladder
}

fn phase_resident_comprehensive_row_retry_ladder(
    preferred_rows: usize,
    minimum_rows: usize,
    absolute_rows: usize,
) -> Vec<usize> {
    let ceiling = preferred_rows.min(absolute_rows);
    [32, 16, 8]
        .into_iter()
        .filter(|&rows| rows <= ceiling && rows >= minimum_rows)
        .collect()
}

struct FrozenPhaseResidentTarget {
    node_name: String,
    frozen_bound: BoundedTensor,
    role: RootIntermediateSweepTargetRole,
}

fn all_target_rows(bounds: &BoundedTensor, deadline: Instant) -> Option<Arc<[u32]>> {
    if !finite_ordered_box(bounds, deadline) {
        return None;
    }
    let mut rows = Vec::new();
    rows.try_reserve_exact(bounds.len()).ok()?;
    for index in 0..bounds.len() {
        if index.is_multiple_of(TRANSCRIPT_POLL_STRIDE) && !deadline_live(deadline) {
            return None;
        }
        rows.push(u32::try_from(index).ok()?);
    }
    deadline_live(deadline).then(|| Arc::from(rows))
}

#[allow(clippy::too_many_arguments)]
fn freeze_phase_resident_targets(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    deadline: Instant,
    min_comprehensive_dim: usize,
    max_comprehensive_dim: usize,
    max_comprehensive_targets: usize,
    max_dense_rows: usize,
) -> Option<Vec<FrozenPhaseResidentTarget>> {
    if max_dense_rows == 0 || !finite_ordered_box(input, deadline) {
        return None;
    }
    let order = graph.exec_order().ok()?.to_vec();
    let dense_names = graph.fc_head_preactivation_targets(&order);
    // Every valid target has at least one row. Refuse an impossible target
    // count before reserving either census, then validate the complete row sum
    // through borrowed live bounds before deep-cloning any tensor.
    if dense_names.is_empty() || dense_names.len() > max_dense_rows {
        return None;
    }
    let mut dense_set = HashSet::new();
    dense_set.try_reserve(dense_names.len()).ok()?;
    let mut dense_borrowed = Vec::new();
    dense_borrowed.try_reserve_exact(dense_names.len()).ok()?;
    let mut dense_rows = 0usize;
    for node_name in &dense_names {
        if !deadline_live(deadline) || !dense_set.insert(node_name.as_str()) {
            return None;
        }
        let live_bound = bounds.get(node_name)?;
        if !finite_ordered_box(live_bound, deadline) {
            return None;
        }
        dense_rows = dense_rows.checked_add(live_bound.len())?;
        if dense_rows > max_dense_rows {
            return None;
        }
        dense_borrowed.push((node_name.as_str(), live_bound));
    }
    let mut frozen = Vec::new();
    frozen.try_reserve(dense_borrowed.len()).ok()?;
    for (node_name, frozen_bound) in dense_borrowed {
        frozen.push(FrozenPhaseResidentTarget {
            node_name: node_name.to_owned(),
            frozen_bound: frozen_bound.clone(),
            role: RootIntermediateSweepTargetRole::DenseMandatory,
        });
    }

    // Ask for enough ranked candidates to remove every dense overlap and
    // still detect one comprehensive target beyond the complete-census cap.
    let census_limit = max_comprehensive_targets
        .checked_add(dense_set.len())?
        .checked_add(1)?;
    let ranked = scoped_wide_demanded_sweep_targets_before(
        graph,
        bounds,
        min_comprehensive_dim,
        max_comprehensive_dim,
        census_limit,
        Some(deadline),
    )?;
    let comprehensive: Vec<_> = ranked
        .into_iter()
        .filter(|node_name| !dense_set.contains(node_name.as_str()))
        .collect();
    if comprehensive.len() > max_comprehensive_targets {
        return None;
    }
    frozen.try_reserve(comprehensive.len()).ok()?;
    for node_name in comprehensive {
        if !deadline_live(deadline) {
            return None;
        }
        let frozen_bound = bounds.get(&node_name)?.clone();
        if !finite_ordered_box(&frozen_bound, deadline) {
            return None;
        }
        frozen.push(FrozenPhaseResidentTarget {
            node_name,
            frozen_bound,
            role: RootIntermediateSweepTargetRole::Comprehensive,
        });
    }
    deadline_live(deadline).then_some(frozen)
}

/// Run the default-dark phase-resident dense plus comprehensive transaction.
///
/// Dense targets and every one of their rows are immutable across capacity
/// negotiation. Only the comprehensive common ceiling may move through the
/// fixed 32/16/8 ladder, and only after a clean predispatch decline. Once a
/// backend accepts a request, every error or publication refusal is terminal.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn root_phase_resident_gpu_intermediate_sweep(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    gpu: &dyn GpuCrownBackward,
    deadline: Instant,
    min_comprehensive_dim: usize,
    max_comprehensive_dim: usize,
    max_comprehensive_rows_per_target: usize,
    max_comprehensive_targets: usize,
    max_dense_rows: usize,
    absolute_max_device_bytes: usize,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> RootIntermediateSweepAttempt {
    if !deadline_live(deadline) {
        return RootIntermediateSweepAttempt::Failed;
    }
    if max_comprehensive_rows_per_target == 0
        || max_dense_rows == 0
        || absolute_max_device_bytes == 0
    {
        return RootIntermediateSweepAttempt::CleanDecline;
    }
    if !gpu.provides_sound_gpu_crown() || !gpu.provides_sound_intermediate_sweep() {
        return RootIntermediateSweepAttempt::CleanDecline;
    }
    let Some(resource_policy) = gpu.intermediate_sweep_resource_policy() else {
        return RootIntermediateSweepAttempt::CleanDecline;
    };
    if !resource_policy.is_valid() {
        return RootIntermediateSweepAttempt::CleanDecline;
    }
    let max_device_bytes = resource_policy
        .max_device_bytes
        .min(absolute_max_device_bytes);
    if max_device_bytes == 0 {
        return RootIntermediateSweepAttempt::CleanDecline;
    }
    let Some(frozen) = freeze_phase_resident_targets(
        graph,
        input,
        bounds,
        deadline,
        min_comprehensive_dim,
        max_comprehensive_dim,
        max_comprehensive_targets,
        max_dense_rows,
    ) else {
        return if deadline_live(deadline) {
            RootIntermediateSweepAttempt::CleanDecline
        } else {
            RootIntermediateSweepAttempt::Failed
        };
    };
    let dense_target_count = frozen
        .iter()
        .filter(|target| target.role == RootIntermediateSweepTargetRole::DenseMandatory)
        .count();
    let comprehensive_target_count = frozen.len().saturating_sub(dense_target_count);
    let row_ladder = if comprehensive_target_count == 0 {
        vec![0]
    } else {
        phase_resident_comprehensive_row_retry_ladder(
            resource_policy.preferred_rows_per_target,
            resource_policy.minimum_rows_per_target,
            max_comprehensive_rows_per_target,
        )
    };
    if row_ladder.is_empty() {
        return RootIntermediateSweepAttempt::CleanDecline;
    }

    for comprehensive_row_ceiling in row_ladder {
        if !deadline_live(deadline) {
            return RootIntermediateSweepAttempt::Failed;
        }
        let mut selected = Vec::new();
        if selected.try_reserve_exact(frozen.len()).is_err() {
            return RootIntermediateSweepAttempt::CleanDecline;
        }
        for target in &frozen {
            let selected_rows = match target.role {
                RootIntermediateSweepTargetRole::DenseMandatory => {
                    all_target_rows(&target.frozen_bound, deadline)
                }
                RootIntermediateSweepTargetRole::Comprehensive => select_balanced_tightening_rows(
                    &target.frozen_bound,
                    comprehensive_row_ceiling,
                    deadline,
                ),
            };
            let Some(selected_rows) = selected_rows else {
                return if deadline_live(deadline) {
                    RootIntermediateSweepAttempt::CleanDecline
                } else {
                    RootIntermediateSweepAttempt::Failed
                };
            };
            selected.push(SelectedSweepTarget {
                node_name: target.node_name.clone(),
                selected_rows,
                frozen_bound: target.frozen_bound.clone(),
                role: Some(target.role),
            });
        }
        let Some(prepared) =
            prepare_root_intermediate_sweep(graph, input, bounds, selected, deadline)
        else {
            return if deadline_live(deadline) {
                RootIntermediateSweepAttempt::CleanDecline
            } else {
                RootIntermediateSweepAttempt::Failed
            };
        };
        let request = GpuIntermediateSweepRequest {
            plan: &prepared.plan,
            input_identity_sha256: prepared.input_identity_sha256,
            input_lower: &prepared.input_lower,
            input_upper: &prepared.input_upper,
            deadline,
            max_device_bytes,
        };
        if let Err(error) = request.validate() {
            eprintln!("[root-phase-resident-crown] host request validation failed: {error}");
            return RootIntermediateSweepAttempt::Failed;
        }
        let result = match gpu.crown_backward_gpu_sound_intermediate_sweep(&request) {
            Ok(Some(result)) => result,
            Ok(None) => {
                eprintln!(
                    "[root-phase-resident-crown] backend predispatch-declined \
                     dense_targets={dense_target_count} comprehensive_targets={comprehensive_target_count} \
                     comprehensive_row_ceiling={comprehensive_row_ceiling} total_rows={}",
                    prepared.plan.total_rows,
                );
                continue;
            }
            Err(error) => {
                eprintln!("[root-phase-resident-crown] accepted request failed: {error}");
                return RootIntermediateSweepAttempt::Failed;
            }
        };
        let validated = match result.validate(&request) {
            Ok(validated) => validated,
            Err(error) => {
                eprintln!("[root-phase-resident-crown] result validation failed: {error}");
                return RootIntermediateSweepAttempt::Failed;
            }
        };
        if !gpu.provides_sound_gpu_crown()
            || !gpu.provides_sound_intermediate_sweep()
            || !deadline_live(deadline)
        {
            return RootIntermediateSweepAttempt::Failed;
        }
        let receipt = *validated.receipt();
        let Some(tightened) = publish_validated_role_bound_batch(
            bounds,
            &prepared.targets,
            validated.targets(),
            deadline,
            || gpu.provides_sound_gpu_crown() && gpu.provides_sound_intermediate_sweep(),
        ) else {
            eprintln!("[root-phase-resident-crown] atomic role-bound publication rejected");
            return RootIntermediateSweepAttempt::Failed;
        };
        let role_bindings: Vec<_> = prepared
            .targets
            .iter()
            .map(|target| {
                (
                    target.role.expect("unified preparation binds every role"),
                    target.target_id,
                    target.node_name.as_str(),
                    target.selected_rows.len(),
                )
            })
            .collect();
        eprintln!(
            "[root-phase-resident-crown] dense_targets={dense_target_count} \
             comprehensive_targets={comprehensive_target_count} total_rows={} tightened={} \
             requested_targets={} completed_targets={} requested_rows={} completed_rows={} \
             peak_bytes={} dispatches={} readbacks={} submits={} syncs={} waves={} \
             h2d_bytes={} d2h_bytes={} graph_sha256={} input_sha256={} bounds_sha256={} \
             target_set_sha256={} role_bindings={role_bindings:?}",
            prepared.plan.total_rows,
            tightened,
            receipt.requested_targets,
            receipt.completed_targets,
            receipt.requested_rows,
            receipt.completed_rows,
            receipt.peak_device_bytes,
            receipt.dispatches,
            receipt.readbacks,
            receipt.submits,
            receipt.synchronizations,
            receipt.waves,
            receipt.host_to_device_bytes,
            receipt.device_to_host_bytes,
            digest_hex(&receipt.graph_identity_sha256),
            digest_hex(&receipt.input_identity_sha256),
            digest_hex(&receipt.bounds_identity_sha256),
            digest_hex(&receipt.target_set_identity_sha256),
        );
        return RootIntermediateSweepAttempt::Completed(tightened);
    }
    RootIntermediateSweepAttempt::CleanDecline
}

/// Run one comprehensive all-target sweep from one frozen root snapshot.
///
/// Every retry contains the complete eligible target set and differs only in
/// its common per-target row ceiling. A backend clean decline may therefore
/// downshift capacity without changing target coverage; once any request is
/// accepted, validation and publication are all-or-none and no other route may
/// run from this phase.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn root_comprehensive_gpu_intermediate_sweep(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    gpu: &dyn GpuCrownBackward,
    deadline: Instant,
    min_dim: usize,
    max_dim: usize,
    max_rows_per_target: usize,
    max_targets: usize,
    absolute_max_device_bytes: usize,
    // #interm-row-chunking DELIVERY: the window count is supplied by the CALLER,
    // resolved from the typed preset key `root_comprehensive_gpu_interm_chunks`
    // with the governed `NY_INTERM_ROW_CHUNKS` lever as an A/B override. It is not
    // read here because the scored entry point exports exactly one NY_* variable,
    // so an env-only setting cannot fire in competition however well it measured
    // (crates/ny-cli/tests/measured_gate_delivery.rs).
    max_chunks: usize,
    // #root-objective-directed-rows: per-target, per-neuron influence on the
    // objective ensemble, keyed by the PRE-ACTIVATION node name (what this sweep
    // tightens). Advisory-only: it reorders the bounded row budget and nothing
    // else. Absent or empty => the historical width ordering, byte for byte.
    influence: Option<&HashMap<String, Vec<f32>>>,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> RootIntermediateSweepAttempt {
    if !deadline_live(deadline) {
        return RootIntermediateSweepAttempt::Failed;
    }
    if max_rows_per_target == 0 || max_targets == 0 || absolute_max_device_bytes == 0 {
        return RootIntermediateSweepAttempt::CleanDecline;
    }
    if !gpu.provides_sound_gpu_crown() || !gpu.provides_sound_intermediate_sweep() {
        return RootIntermediateSweepAttempt::CleanDecline;
    }
    let Some(resource_policy) = gpu.intermediate_sweep_resource_policy() else {
        return RootIntermediateSweepAttempt::CleanDecline;
    };
    if !resource_policy.is_valid() {
        return RootIntermediateSweepAttempt::CleanDecline;
    }
    let preferred_rows = resource_policy
        .preferred_rows_per_target
        .min(max_rows_per_target);
    let minimum_rows = resource_policy.minimum_rows_per_target;
    let retry_ladder = comprehensive_row_retry_ladder(preferred_rows, minimum_rows);
    let max_device_bytes = resource_policy
        .max_device_bytes
        .min(absolute_max_device_bytes);
    if retry_ladder.is_empty() || max_device_bytes == 0 {
        return RootIntermediateSweepAttempt::CleanDecline;
    }

    let Some(census_limit) = max_targets.checked_add(1) else {
        return RootIntermediateSweepAttempt::CleanDecline;
    };
    let Some(ranked) = scoped_wide_demanded_sweep_targets_before(
        graph,
        bounds,
        min_dim,
        max_dim,
        census_limit,
        Some(deadline),
    ) else {
        return if deadline_live(deadline) {
            RootIntermediateSweepAttempt::CleanDecline
        } else {
            RootIntermediateSweepAttempt::Failed
        };
    };
    if ranked.is_empty() {
        return RootIntermediateSweepAttempt::CleanDecline;
    }
    if ranked.len() > max_targets {
        eprintln!(
            "[root-comprehensive-gpu-interm-sweep] eligible target census exceeds cap \
             {max_targets}; refusing the whole route"
        );
        return RootIntermediateSweepAttempt::CleanDecline;
    }

    // Freeze every eligible target before the first backend call. The mutable
    // map is not touched again until the single atomic publication below.
    let mut frozen = Vec::new();
    if frozen.try_reserve_exact(ranked.len()).is_err() {
        return RootIntermediateSweepAttempt::CleanDecline;
    }
    for node_name in ranked {
        if !deadline_live(deadline) {
            return RootIntermediateSweepAttempt::Failed;
        }
        let Some(frozen_bound) = bounds.get(&node_name) else {
            return RootIntermediateSweepAttempt::CleanDecline;
        };
        if select_balanced_tightening_rows(frozen_bound, minimum_rows, deadline).is_none() {
            return if deadline_live(deadline) {
                RootIntermediateSweepAttempt::CleanDecline
            } else {
                RootIntermediateSweepAttempt::Failed
            };
        }
        frozen.push((node_name, frozen_bound.clone()));
    }

    // #interm-row-chunking: the bounded sweep is hard-capped at ~128 rows/target
    // by device memory (the backend declines 256 outright), which is ~2.8%
    // coverage, and the measured benefit is LINEAR in coverage (0.026
    // unstable-flips/row at 1,152 rows vs 0.021 for the full-coverage CPU pass).
    // So a SINGLE bounded sweep can never reach the coverage the root needs, no
    // matter how the ceiling is tuned. Instead run the same bounded sweep
    // repeatedly over disjoint row windows and accumulate, which keeps peak
    // device memory at exactly one chunk's worth while coverage grows with the
    // time available.
    //
    // Sound by construction, no new bound math: each sweep is atomic over its own
    // window, each commit is a shrink-only intersect into the live map, and all
    // windows are cut from ONE frozen transcript so they stay disjoint and stable
    // even as earlier chunks tighten the live bounds. Stopping early — deadline,
    // exhaustion, or a mid-run decline — simply leaves fewer chunks applied, and
    // every applied chunk is independently valid.
    //
    // The window count arrives from the caller (preset key, or the governed
    // `NY_INTERM_ROW_CHUNKS` lever as an override). `0` means the shipped single
    // sweep to this reader, not "no sweep", so it is clamped rather than refused.
    let max_chunks = max_chunks.max(1);
    let mut chunk_skip = 0usize;
    let mut chunks_done = 0usize;
    let mut accumulated_tightened = 0usize;
    let mut exhausted = false;

    // Index-driven ladder rather than a `for`: a backend decline advances to the
    // next (smaller) rung as before, but a SUCCESS may repeat the same rung for
    // the next disjoint row window.
    let ladder = retry_ladder;
    let mut ladder_index = 0usize;
    while let Some(&row_limit) = ladder.get(ladder_index) {
        if !deadline_live(deadline) {
            return RootIntermediateSweepAttempt::Failed;
        }
        // #comprehensive-rows-probe: the receipt reports peak bytes, dispatches and
        // waves but no WALL TIME, so the sweep's cost per row is unmeasurable from
        // its own output. Row coverage is the binding design question here (16
        // rows/target is 0.26% of the eligible neurons), and choosing between one
        // wide sweep and row-chunked accumulation needs seconds-per-row, not just
        // bytes-per-row. Print-only.
        let attempt_t0 = Instant::now();
        let mut selected = Vec::new();
        if selected.try_reserve_exact(frozen.len()).is_err() {
            return RootIntermediateSweepAttempt::CleanDecline;
        }
        for (node_name, frozen_bound) in &frozen {
            // #interm-row-chunking: take the window `[chunk_skip, +row_limit)` of
            // the balanced ordering rather than always its head, so successive
            // chunks cover DISJOINT rows. `chunk_skip == 0` on the first pass is
            // byte-identical to the historical head-only selection.
            //
            // A `None` here is no longer automatically a decline: once at least
            // one chunk has been committed, running off the end of a target's
            // candidate list is normal EXHAUSTION, and everything already
            // committed stays valid (each sweep is atomic, each commit is
            // shrink-only). Only a first-chunk `None` is a real decline.
            // A target with no influence entry, or one whose entries are all
            // zero, falls back to width: an all-zero key would collapse the
            // ordering to index order, which is strictly worse than width.
            let node_influence = influence
                .and_then(|map| map.get(node_name))
                .filter(|weights| weights.iter().any(|w| *w != 0.0))
                .map(Vec::as_slice);
            let selected_rows = select_balanced_tightening_rows_chunk(
                frozen_bound,
                chunk_skip,
                row_limit,
                deadline,
                node_influence,
            );
            let Some(selected_rows) = selected_rows else {
                if chunk_skip > 0 {
                    exhausted = true;
                    break;
                }
                return if deadline_live(deadline) {
                    RootIntermediateSweepAttempt::CleanDecline
                } else {
                    RootIntermediateSweepAttempt::Failed
                };
            };
            selected.push(SelectedSweepTarget {
                node_name: node_name.clone(),
                selected_rows,
                frozen_bound: frozen_bound.clone(),
                role: None,
            });
        }
        let Some(prepared) =
            prepare_root_intermediate_sweep(graph, input, bounds, selected, deadline)
        else {
            return if deadline_live(deadline) {
                eprintln!(
                    "[root-comprehensive-gpu-interm-sweep] complete target set is structurally \
                     unpreparable; refusing the whole route"
                );
                RootIntermediateSweepAttempt::CleanDecline
            } else {
                RootIntermediateSweepAttempt::Failed
            };
        };
        let request = GpuIntermediateSweepRequest {
            plan: &prepared.plan,
            input_identity_sha256: prepared.input_identity_sha256,
            input_lower: &prepared.input_lower,
            input_upper: &prepared.input_upper,
            deadline,
            max_device_bytes,
        };
        if let Err(error) = request.validate() {
            eprintln!(
                "[root-comprehensive-gpu-interm-sweep] host request validation failed: {error}"
            );
            return RootIntermediateSweepAttempt::Failed;
        }
        let result = match gpu.crown_backward_gpu_sound_intermediate_sweep(&request) {
            Ok(Some(result)) => result,
            Ok(None) => {
                eprintln!(
                    "[root-comprehensive-gpu-interm-sweep] backend predispatch-declined \
                     targets={} row_ceiling={row_limit} total_rows={}",
                    prepared.targets.len(),
                    prepared.plan.total_rows,
                );
                ladder_index += 1;
                continue;
            }
            Err(error) => {
                eprintln!("[root-comprehensive-gpu-interm-sweep] accepted request failed: {error}");
                return RootIntermediateSweepAttempt::Failed;
            }
        };
        let validated = match result.validate(&request) {
            Ok(validated) => validated,
            Err(error) => {
                eprintln!(
                    "[root-comprehensive-gpu-interm-sweep] result validation failed: {error}"
                );
                return RootIntermediateSweepAttempt::Failed;
            }
        };
        if !gpu.provides_sound_gpu_crown()
            || !gpu.provides_sound_intermediate_sweep()
            || !deadline_live(deadline)
        {
            return RootIntermediateSweepAttempt::Failed;
        }
        let receipt = *validated.receipt();
        let Some(tightened) = publish_validated_batch(
            bounds,
            &prepared.targets,
            validated.targets(),
            deadline,
            || gpu.provides_sound_gpu_crown() && gpu.provides_sound_intermediate_sweep(),
        ) else {
            eprintln!(
                "[root-comprehensive-gpu-interm-sweep] atomic all-target publication rejected"
            );
            return RootIntermediateSweepAttempt::Failed;
        };
        let target_bindings: Vec<_> = prepared
            .targets
            .iter()
            .map(|target| {
                (
                    target.target_id,
                    target.node_name.as_str(),
                    target.selected_rows.as_ref(),
                )
            })
            .collect();
        eprintln!(
            "[root-comprehensive-gpu-interm-sweep] elapsed={:.2}s targets={} row_ceiling={row_limit} \
             rows={} tightened={} requested_targets={} completed_targets={} \
             requested_rows={} completed_rows={} peak_bytes={} dispatches={} \
             readbacks={} submits={} syncs={} waves={} h2d_bytes={} d2h_bytes={} \
             graph_sha256={} input_sha256={} bounds_sha256={} target_set_sha256={} \
             target_bindings={target_bindings:?}",
            attempt_t0.elapsed().as_secs_f64(),
            prepared.targets.len(),
            prepared.plan.total_rows,
            tightened,
            receipt.requested_targets,
            receipt.completed_targets,
            receipt.requested_rows,
            receipt.completed_rows,
            receipt.peak_device_bytes,
            receipt.dispatches,
            receipt.readbacks,
            receipt.submits,
            receipt.synchronizations,
            receipt.waves,
            receipt.host_to_device_bytes,
            receipt.device_to_host_bytes,
            digest_hex(&receipt.graph_identity_sha256),
            digest_hex(&receipt.input_identity_sha256),
            digest_hex(&receipt.bounds_identity_sha256),
            digest_hex(&receipt.target_set_identity_sha256),
        );
        // #interm-row-chunking: accumulate and advance to the next disjoint row
        // window at the SAME rung. Every committed chunk is independently valid,
        // so any stop condition returns what has already been applied.
        accumulated_tightened = accumulated_tightened.saturating_add(tightened);
        chunks_done += 1;
        chunk_skip = chunk_skip.saturating_add(row_limit);
        if exhausted || chunks_done >= max_chunks || !deadline_live(deadline) {
            if max_chunks > 1 {
                eprintln!(
                    "[root-comprehensive-gpu-interm-sweep] chunking done \
                     chunks={chunks_done}/{max_chunks} rows_covered={chunk_skip}/target \
                     exhausted={exhausted} tightened_total={accumulated_tightened}"
                );
            }
            return RootIntermediateSweepAttempt::Completed(accumulated_tightened);
        }
        continue;
    }
    RootIntermediateSweepAttempt::CleanDecline
}

#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn root_wide_demanded_intermediate_sweep(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    gpu: &dyn GpuCrownBackward,
    deadline: Instant,
    min_dim: usize,
    max_dim: usize,
    max_rows: usize,
    max_targets: usize,
    max_preflights: usize,
    max_device_bytes: usize,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> RootIntermediateSweepAttempt {
    // The builder is multi-target-ready, but the first production slice is one
    // target per atomic request. Refuse a future policy widening until its
    // selection/commit scheduling receives its own measured review.
    if !deadline_live(deadline) {
        return RootIntermediateSweepAttempt::Failed;
    }
    if max_targets != 1 || max_rows == 0 || max_preflights == 0 || max_device_bytes == 0 {
        return RootIntermediateSweepAttempt::CleanDecline;
    }
    if !gpu.provides_sound_gpu_crown() || !gpu.provides_sound_intermediate_sweep() {
        return RootIntermediateSweepAttempt::CleanDecline;
    }
    let Some(ranked) = scoped_wide_demanded_sweep_targets_before(
        graph,
        bounds,
        min_dim,
        max_dim,
        max_preflights,
        Some(deadline),
    ) else {
        return if deadline_live(deadline) {
            RootIntermediateSweepAttempt::CleanDecline
        } else {
            RootIntermediateSweepAttempt::Failed
        };
    };

    for node_name in ranked.into_iter().take(max_preflights) {
        if !deadline_live(deadline) {
            return RootIntermediateSweepAttempt::Failed;
        }
        let Some(frozen_bound) = bounds.get(&node_name) else {
            continue;
        };
        let Some(full_selection) = select_tightening_rows(frozen_bound, max_rows, deadline) else {
            if !deadline_live(deadline) {
                return RootIntermediateSweepAttempt::Failed;
            }
            continue;
        };
        for row_limit in row_retry_ladder(full_selection.len()) {
            if !deadline_live(deadline) {
                return RootIntermediateSweepAttempt::Failed;
            }
            let selected_rows = if row_limit == full_selection.len() {
                Arc::clone(&full_selection)
            } else {
                let Some(rows) = select_tightening_rows(frozen_bound, row_limit, deadline) else {
                    return if deadline_live(deadline) {
                        RootIntermediateSweepAttempt::CleanDecline
                    } else {
                        RootIntermediateSweepAttempt::Failed
                    };
                };
                rows
            };
            let selection = SelectedSweepTarget {
                node_name: node_name.clone(),
                selected_rows,
                frozen_bound: frozen_bound.clone(),
                role: None,
            };
            let Some(prepared) =
                prepare_root_intermediate_sweep(graph, input, bounds, vec![selection], deadline)
            else {
                if !deadline_live(deadline) {
                    return RootIntermediateSweepAttempt::Failed;
                }
                eprintln!(
                    "[root-wide-demanded-interm-sweep] preflight target='{node_name}' rows={row_limit} declined"
                );
                break;
            };
            let request = GpuIntermediateSweepRequest {
                plan: &prepared.plan,
                input_identity_sha256: prepared.input_identity_sha256,
                input_lower: &prepared.input_lower,
                input_upper: &prepared.input_upper,
                deadline,
                max_device_bytes,
            };
            if let Err(error) = request.validate() {
                eprintln!(
                    "[root-wide-demanded-interm-sweep] host request validation failed: {error}"
                );
                return RootIntermediateSweepAttempt::Failed;
            }
            let result = match gpu.crown_backward_gpu_sound_intermediate_sweep(&request) {
                Ok(Some(result)) => result,
                Ok(None) => {
                    eprintln!(
                        "[root-wide-demanded-interm-sweep] backend predispatch-declined \
                         target='{node_name}' rows={row_limit}"
                    );
                    continue;
                }
                Err(error) => {
                    eprintln!(
                        "[root-wide-demanded-interm-sweep] accepted request failed for \
                         target='{node_name}' rows={row_limit}: {error}"
                    );
                    return RootIntermediateSweepAttempt::Failed;
                }
            };
            let validated = match result.validate(&request) {
                Ok(validated) => validated,
                Err(error) => {
                    eprintln!(
                        "[root-wide-demanded-interm-sweep] result validation failed for \
                         target='{node_name}' rows={row_limit}: {error}"
                    );
                    return RootIntermediateSweepAttempt::Failed;
                }
            };
            if !gpu.provides_sound_gpu_crown()
                || !gpu.provides_sound_intermediate_sweep()
                || !deadline_live(deadline)
            {
                return RootIntermediateSweepAttempt::Failed;
            }
            let receipt = *validated.receipt();
            let Some(tightened) = publish_validated_batch(
                bounds,
                &prepared.targets,
                validated.targets(),
                deadline,
                || gpu.provides_sound_gpu_crown() && gpu.provides_sound_intermediate_sweep(),
            ) else {
                eprintln!(
                    "[root-wide-demanded-interm-sweep] atomic publication rejected target='{node_name}'"
                );
                return RootIntermediateSweepAttempt::Failed;
            };
            eprintln!(
                "[root-wide-demanded-interm-sweep] target='{node_name}' rows={} tightened={} \
                 graph={:016x} bounds={:016x} targets={:016x} input={:016x} \
                 peak_bytes={} dispatches={} readbacks={} submits={} syncs={} waves={}",
                prepared.plan.total_rows,
                tightened,
                u64::from_le_bytes(receipt.graph_identity_sha256[..8].try_into().unwrap()),
                u64::from_le_bytes(receipt.bounds_identity_sha256[..8].try_into().unwrap()),
                u64::from_le_bytes(receipt.target_set_identity_sha256[..8].try_into().unwrap()),
                u64::from_le_bytes(receipt.input_identity_sha256[..8].try_into().unwrap()),
                receipt.peak_device_bytes,
                receipt.dispatches,
                receipt.readbacks,
                receipt.submits,
                receipt.synchronizations,
                receipt.waves,
            );
            return RootIntermediateSweepAttempt::Completed(tightened);
        }
    }
    RootIntermediateSweepAttempt::CleanDecline
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use ndarray::{arr1, arr2};
    use ny_tensor::L2Constraint;

    use super::*;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer, ReshapeLayer};
    use crate::GraphNode;

    #[derive(Clone, Debug)]
    struct RecordedRequest {
        rows: usize,
        targets: usize,
        target_rows: Vec<(u64, Vec<u32>)>,
        max_device_bytes: usize,
        graph_identity_sha256: [u8; 32],
        bounds_identity_sha256: [u8; 32],
        target_set_identity_sha256: [u8; 32],
        input_identity_sha256: [u8; 32],
    }

    struct DeclineUntilSmallBackend {
        accept_at_rows: usize,
        preferred_rows: usize,
        poison_last_target: bool,
        corrupt_target_identity: bool,
        attempts: Mutex<Vec<RecordedRequest>>,
        accepted: Mutex<usize>,
    }

    impl DeclineUntilSmallBackend {
        fn new(accept_at_rows: usize) -> Self {
            Self {
                accept_at_rows,
                preferred_rows: 8,
                poison_last_target: false,
                corrupt_target_identity: false,
                attempts: Mutex::new(Vec::new()),
                accepted: Mutex::new(0),
            }
        }

        fn poison_last_target(accept_at_rows: usize) -> Self {
            Self {
                accept_at_rows,
                preferred_rows: 8,
                poison_last_target: true,
                corrupt_target_identity: false,
                attempts: Mutex::new(Vec::new()),
                accepted: Mutex::new(0),
            }
        }

        fn with_preferred_rows(mut self, preferred_rows: usize) -> Self {
            self.preferred_rows = preferred_rows;
            self
        }

        fn with_corrupt_target_identity(mut self) -> Self {
            self.corrupt_target_identity = true;
            self
        }
    }

    impl GpuCrownBackward for DeclineUntilSmallBackend {
        fn provides_sound_gpu_crown(&self) -> bool {
            true
        }

        fn provides_sound_intermediate_sweep(&self) -> bool {
            true
        }

        fn intermediate_sweep_resource_policy(
            &self,
        ) -> Option<ny_core::GpuIntermediateSweepResourcePolicy> {
            Some(ny_core::GpuIntermediateSweepResourcePolicy {
                max_device_bytes: 512 * 1024 * 1024,
                preferred_rows_per_target: self.preferred_rows,
                minimum_rows_per_target: 8,
            })
        }

        fn crown_backward_gpu_sound_intermediate_sweep(
            &self,
            request: &GpuIntermediateSweepRequest<'_>,
        ) -> ny_core::Result<Option<ny_core::GpuIntermediateSweepResult>> {
            request.validate()?;
            self.attempts
                .lock()
                .expect("attempt lock")
                .push(RecordedRequest {
                    rows: request.plan.total_rows,
                    targets: request.plan.injections.len(),
                    target_rows: request
                        .plan
                        .injections
                        .iter()
                        .map(|injection| {
                            (
                                injection.target_id,
                                injection.selected_rows.as_ref().to_vec(),
                            )
                        })
                        .collect(),
                    max_device_bytes: request.max_device_bytes,
                    graph_identity_sha256: request.plan.graph_identity_sha256,
                    bounds_identity_sha256: request.plan.bounds_identity_sha256,
                    target_set_identity_sha256: request.plan.target_set_identity_sha256,
                    input_identity_sha256: request.input_identity_sha256,
                });
            if request.plan.total_rows > self.accept_at_rows {
                return Ok(None);
            }

            let mut accepted = self.accepted.lock().expect("accepted lock");
            *accepted += 1;
            let target_count = request.plan.injections.len();
            let targets = request
                .plan
                .injections
                .iter()
                .enumerate()
                .map(|(index, injection)| {
                    let poison = self.poison_last_target && index + 1 == target_count;
                    ny_core::GpuIntermediateTargetResult {
                        target_id: injection.target_id,
                        row_offset: injection.row_offset,
                        selected_rows: Arc::clone(&injection.selected_rows),
                        lower_bounds: vec![
                            if poison { 100.0 } else { 1.25 };
                            injection.selected_rows.len()
                        ],
                        upper_bounds: vec![
                            if poison { 101.0 } else { 2.75 };
                            injection.selected_rows.len()
                        ],
                    }
                })
                .collect();
            let mut target_set_identity_sha256 = request.plan.target_set_identity_sha256;
            if self.corrupt_target_identity {
                target_set_identity_sha256[0] ^= 1;
            }
            let receipt = ny_core::GpuIntermediateSweepReceipt {
                graph_identity_sha256: request.plan.graph_identity_sha256,
                input_identity_sha256: request.input_identity_sha256,
                bounds_identity_sha256: request.plan.bounds_identity_sha256,
                target_set_identity_sha256,
                requested_targets: request.plan.injections.len(),
                completed_targets: request.plan.injections.len(),
                requested_rows: request.plan.total_rows,
                completed_rows: request.plan.total_rows,
                peak_device_bytes: 1,
                dispatches: 1,
                host_to_device_bytes: 0,
                device_to_host_bytes: request.plan.total_rows * 2 * size_of::<f32>(),
                readbacks: 1,
                submits: 1,
                synchronizations: 1,
                waves: 1,
            };
            Ok(Some(ny_core::GpuIntermediateSweepResult::new_unvalidated(
                targets, receipt,
            )))
        }

        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> ny_core::Result<ny_core::GpuCrownResult> {
            Err(NyError::UnsupportedOp("retry-only test backend".into()))
        }
    }

    fn bounded(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn()).expect("valid box")
    }

    fn selected(
        name: &str,
        rows: &[u32],
        bounds: &HashMap<String, BoundedTensor>,
    ) -> SelectedSweepTarget {
        SelectedSweepTarget {
            node_name: name.to_string(),
            selected_rows: Arc::from(rows),
            frozen_bound: bounds.get(name).expect("selected target bound").clone(),
            role: None,
        }
    }

    fn linear(
        name: &str,
        input: &str,
        weight: ndarray::Array2<f32>,
        bias: ndarray::Array1<f32>,
    ) -> GraphNode {
        let layer = Layer::Linear(LinearLayer::new(weight, Some(bias)).expect("linear"));
        if input == NETWORK_INPUT {
            GraphNode::from_input(name, layer)
        } else {
            GraphNode::new(name, layer, vec![input.to_string()])
        }
    }

    fn chain_fixture() -> (GraphNetwork, HashMap<String, BoundedTensor>, BoundedTensor) {
        let mut graph = GraphNetwork::new();
        graph.add_node(linear(
            "linear0",
            NETWORK_INPUT,
            arr2(&[[1.0, -0.5], [-1.5, 0.25], [0.75, 2.0]]),
            arr1(&[0.1, -0.2, 0.3]),
        ));
        graph.add_node(GraphNode::new(
            "reshape",
            Layer::Reshape(ReshapeLayer::new(vec![3])),
            vec!["linear0".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu0",
            Layer::ReLU(ReLULayer),
            vec!["reshape".into()],
        ));
        graph.add_node(linear(
            "linear1",
            "relu0",
            arr2(&[[0.5, -1.0, 0.25], [1.25, 0.75, -0.5]]),
            arr1(&[-0.1, 0.2]),
        ));
        graph.set_output("linear1");
        let bounds = HashMap::from([
            (
                "linear0".into(),
                bounded(&[-2.0, -2.0, -3.0], &[2.0, 2.0, 3.0]),
            ),
            (
                "reshape".into(),
                bounded(&[-2.0, -2.0, -3.0], &[2.0, 2.0, 3.0]),
            ),
            ("relu0".into(), bounded(&[0.0, 0.0, 0.0], &[2.0, 2.0, 3.0])),
            ("linear1".into(), bounded(&[-4.0, -4.0], &[4.0, 4.0])),
        ]);
        (graph, bounds, bounded(&[-1.0, -0.5], &[1.0, 0.75]))
    }

    fn two_target_residual_fixture(
        dimension: usize,
    ) -> (GraphNetwork, HashMap<String, BoundedTensor>, BoundedTensor) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "stem",
            Layer::Reshape(ReshapeLayer::new(vec![dimension as i64])),
        ));
        graph.add_node(GraphNode::new(
            "wide0",
            Layer::Add(AddLayer),
            vec!["stem".into(), "stem".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu0",
            Layer::ReLU(ReLULayer),
            vec!["wide0".into()],
        ));
        graph.add_node(GraphNode::new(
            "wide1",
            Layer::Add(AddLayer),
            vec!["relu0".into(), "stem".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["wide1".into()],
        ));
        graph.set_output("relu1");
        graph
            .exec_order()
            .expect("cache deterministic execution order");

        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[dimension]), -1.0),
            ArrayD::from_elem(IxDyn(&[dimension]), 1.0),
        )
        .expect("input box");
        let bounds = HashMap::from([
            ("stem".into(), input.clone()),
            (
                "wide0".into(),
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[dimension]), -2.0),
                    ArrayD::from_elem(IxDyn(&[dimension]), 2.0),
                )
                .expect("wide0 box"),
            ),
            (
                "relu0".into(),
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[dimension]), 0.0),
                    ArrayD::from_elem(IxDyn(&[dimension]), 2.0),
                )
                .expect("relu0 box"),
            ),
            (
                "wide1".into(),
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[dimension]), -4.0),
                    ArrayD::from_elem(IxDyn(&[dimension]), 4.0),
                )
                .expect("wide1 box"),
            ),
            (
                "relu1".into(),
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[dimension]), 0.0),
                    ArrayD::from_elem(IxDyn(&[dimension]), 4.0),
                )
                .expect("relu1 box"),
            ),
        ]);
        (graph, bounds, input)
    }

    fn phase_resident_fixture(
        comprehensive_dim: usize,
        dense_dim: usize,
    ) -> (GraphNetwork, HashMap<String, BoundedTensor>, BoundedTensor) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "stem",
            Layer::Reshape(ReshapeLayer::new(vec![comprehensive_dim as i64])),
        ));
        graph.add_node(GraphNode::new(
            "wide",
            Layer::Add(AddLayer),
            vec!["stem".into(), "stem".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu_wide",
            Layer::ReLU(ReLULayer),
            vec!["wide".into()],
        ));
        let weight =
            ndarray::Array2::from_shape_fn((dense_dim, comprehensive_dim), |(row, column)| {
                if column % dense_dim == row {
                    0.5
                } else {
                    0.0
                }
            });
        graph.add_node(linear(
            "head",
            "relu_wide",
            weight,
            ndarray::Array1::zeros(dense_dim),
        ));
        graph.add_node(GraphNode::new(
            "relu_head",
            Layer::ReLU(ReLULayer),
            vec!["head".into()],
        ));
        graph.set_output("relu_head");
        graph
            .exec_order()
            .expect("cache deterministic execution order");

        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[comprehensive_dim]), -1.0),
            ArrayD::from_elem(IxDyn(&[comprehensive_dim]), 1.0),
        )
        .expect("input box");
        let bounds = HashMap::from([
            ("stem".into(), input.clone()),
            (
                "wide".into(),
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[comprehensive_dim]), -2.0),
                    ArrayD::from_elem(IxDyn(&[comprehensive_dim]), 2.0),
                )
                .expect("wide box"),
            ),
            (
                "relu_wide".into(),
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[comprehensive_dim]), 0.0),
                    ArrayD::from_elem(IxDyn(&[comprehensive_dim]), 2.0),
                )
                .expect("wide relu box"),
            ),
            (
                "head".into(),
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[dense_dim]), -2.0),
                    ArrayD::from_elem(IxDyn(&[dense_dim]), 2.0),
                )
                .expect("head box"),
            ),
            (
                "relu_head".into(),
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[dense_dim]), 0.0),
                    ArrayD::from_elem(IxDyn(&[dense_dim]), 2.0),
                )
                .expect("head relu box"),
            ),
        ]);
        (graph, bounds, input)
    }

    #[test]
    fn row_selection_prioritizes_crossings_then_stable_nonpoints_and_canonicalizes_ids() {
        let box_ = bounded(&[1.0, -1.0, -20.0, -4.0, 2.0], &[11.0, 1.0, -5.0, 3.0, 2.0]);
        let rows = select_tightening_rows(&box_, 3, Instant::now() + Duration::from_secs(2))
            .expect("crossing plus stable rows");
        assert_eq!(rows.as_ref(), &[1, 2, 3]);
        assert!(
            rows.contains(&1) && rows.contains(&3),
            "both crossings come first"
        );
        assert!(
            rows.contains(&2),
            "widest stable non-point fills the remainder"
        );
        assert!(!rows.contains(&0), "narrower stable row is capped");
        assert!(!rows.contains(&4), "point rows are never requested");

        let stable = bounded(&[-9.0, 2.0, 4.0], &[-2.0, 8.0, 4.0]);
        let rows = select_tightening_rows(&stable, 2, Instant::now() + Duration::from_secs(2))
            .expect("zero-crossing target remains useful");
        assert_eq!(rows.as_ref(), &[0, 1]);
    }

    #[test]
    fn balanced_row_selection_retains_crossing_and_stable_samples() {
        let box_ = bounded(
            &[
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0,
            ],
        );
        let rows =
            select_balanced_tightening_rows(&box_, 8, Instant::now() + Duration::from_secs(2))
                .expect("balanced rows");
        assert_eq!(rows.as_ref(), &[0, 1, 2, 3, 8, 9, 10, 11]);
    }

    #[test]
    fn row_retry_ladder_halves_to_the_bounded_floor() {
        assert_eq!(row_retry_ladder(512), vec![512, 256, 128, 64, 32]);
        assert_eq!(row_retry_ladder(33), vec![33, 32]);
        assert_eq!(row_retry_ladder(17), vec![17]);
        assert!(row_retry_ladder(0).is_empty());
        assert_eq!(comprehensive_row_retry_ladder(32, 8), vec![32, 16, 8]);
        assert_eq!(comprehensive_row_retry_ladder(16, 8), vec![16, 8]);
        assert!(comprehensive_row_retry_ladder(4, 8).is_empty());
        assert_eq!(
            phase_resident_comprehensive_row_retry_ladder(32, 8, 32),
            vec![32, 16, 8]
        );
        assert_eq!(
            phase_resident_comprehensive_row_retry_ladder(16, 8, 32),
            vec![16, 8]
        );
        assert_eq!(
            phase_resident_comprehensive_row_retry_ladder(32, 16, 32),
            vec![32, 16]
        );
        assert!(phase_resident_comprehensive_row_retry_ladder(8, 16, 32).is_empty());
    }

    #[test]
    fn sealed_receipt_digest_is_complete_fixed_width_hex() {
        let digest = std::array::from_fn(|index| index as u8);
        assert_eq!(
            digest_hex(&digest),
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
        );
    }

    #[test]
    fn comprehensive_route_sends_two_targets_once_and_commits_them_atomically() {
        let (graph, mut bounds, input) = two_target_residual_fixture(8);
        let backend = DeclineUntilSmallBackend::new(16);

        assert_eq!(
            root_comprehensive_gpu_intermediate_sweep(
                &graph,
                &input,
                &backend,
                Instant::now() + Duration::from_secs(5),
                8,
                8,
                32,
                4,
                12 * 1024 * 1024 * 1024,
                // 1 window: the historical single sweep these fixtures assert,
                // byte-identical to the pre-chunking behaviour.
                1,
                // influence: None => the historical width ordering, byte for
                // byte, which is the ordering these tests pin.
                None,
                &mut bounds,
            ),
            RootIntermediateSweepAttempt::Completed(2)
        );
        let attempts = backend.attempts.lock().expect("attempt lock");
        assert_eq!(attempts.len(), 1, "all targets share one backend call");
        assert_eq!(attempts[0].targets, 2);
        assert_eq!(attempts[0].rows, 16);
        assert_eq!(*backend.accepted.lock().expect("accepted lock"), 1);
        assert!(bounds["wide0"].lower().iter().all(|&value| value == 1.25));
        assert!(bounds["wide1"].lower().iter().all(|&value| value == 1.25));
    }

    #[test]
    fn comprehensive_route_rejects_one_bad_target_without_publishing_a_prefix() {
        let (graph, mut bounds, input) = two_target_residual_fixture(8);
        let before = map_bits(&bounds);
        let backend = DeclineUntilSmallBackend::poison_last_target(16);

        assert_eq!(
            root_comprehensive_gpu_intermediate_sweep(
                &graph,
                &input,
                &backend,
                Instant::now() + Duration::from_secs(5),
                8,
                8,
                32,
                4,
                12 * 1024 * 1024 * 1024,
                // 1 window: the historical single sweep these fixtures assert,
                // byte-identical to the pre-chunking behaviour.
                1,
                // influence: None => the historical width ordering, byte for
                // byte, which is the ordering these tests pin.
                None,
                &mut bounds,
            ),
            RootIntermediateSweepAttempt::Failed
        );
        assert_eq!(backend.attempts.lock().expect("attempt lock").len(), 1);
        assert_eq!(map_bits(&bounds), before);
    }

    #[test]
    fn comprehensive_capacity_retry_keeps_every_target_in_every_transcript() {
        let (graph, mut bounds, input) = two_target_residual_fixture(16);
        let backend = DeclineUntilSmallBackend::new(16).with_preferred_rows(16);

        assert_eq!(
            root_comprehensive_gpu_intermediate_sweep(
                &graph,
                &input,
                &backend,
                Instant::now() + Duration::from_secs(5),
                16,
                16,
                32,
                4,
                12 * 1024 * 1024 * 1024,
                // 1 window: the historical single sweep these fixtures assert,
                // byte-identical to the pre-chunking behaviour.
                1,
                // influence: None => the historical width ordering, byte for
                // byte, which is the ordering these tests pin.
                None,
                &mut bounds,
            ),
            RootIntermediateSweepAttempt::Completed(2)
        );
        let attempts = backend.attempts.lock().expect("attempt lock");
        assert_eq!(attempts.len(), 2);
        assert_eq!(
            attempts
                .iter()
                .map(|attempt| (attempt.targets, attempt.rows))
                .collect::<Vec<_>>(),
            [(2, 32), (2, 16)]
        );
        assert_eq!(
            attempts[0].graph_identity_sha256,
            attempts[1].graph_identity_sha256
        );
        assert_eq!(
            attempts[0].bounds_identity_sha256,
            attempts[1].bounds_identity_sha256
        );
        assert_eq!(
            attempts[0].input_identity_sha256,
            attempts[1].input_identity_sha256
        );
        assert_ne!(
            attempts[0].target_set_identity_sha256,
            attempts[1].target_set_identity_sha256
        );
    }

    #[test]
    fn phase_resident_retry_changes_only_comprehensive_rows() {
        let (graph, mut bounds, input) = phase_resident_fixture(32, 4);
        let backend = DeclineUntilSmallBackend::new(12).with_preferred_rows(32);

        assert_eq!(
            root_phase_resident_gpu_intermediate_sweep(
                &graph,
                &input,
                &backend,
                Instant::now() + Duration::from_secs(5),
                32,
                32,
                32,
                4,
                512,
                8 * 1024 * 1024 * 1024,
                &mut bounds,
            ),
            RootIntermediateSweepAttempt::Completed(2)
        );
        let attempts = backend.attempts.lock().expect("attempt lock");
        assert_eq!(attempts.len(), 3);
        assert_eq!(
            attempts
                .iter()
                .map(|attempt| (attempt.targets, attempt.rows))
                .collect::<Vec<_>>(),
            [(2, 36), (2, 20), (2, 12)]
        );
        let dense_target = attempts[0]
            .target_rows
            .iter()
            .find(|(_, rows)| rows.len() == 4)
            .map(|(target, _)| *target)
            .expect("one mandatory four-row dense target");
        for attempt in attempts.iter() {
            let dense_rows = attempt
                .target_rows
                .iter()
                .find(|(target, _)| *target == dense_target)
                .map(|(_, rows)| rows.as_slice())
                .expect("dense target remains in every request");
            assert_eq!(dense_rows, &[0, 1, 2, 3]);
            assert_eq!(attempt.targets, 2, "the complete census cannot shrink");
            assert_eq!(attempt.max_device_bytes, 512 * 1024 * 1024);
        }
        let comprehensive_rows: Vec<_> = attempts
            .iter()
            .map(|attempt| {
                attempt
                    .target_rows
                    .iter()
                    .find(|(target, _)| *target != dense_target)
                    .map(|(_, rows)| rows.len())
                    .expect("comprehensive target remains in every request")
            })
            .collect();
        assert_eq!(comprehensive_rows, [32, 16, 8]);
        assert!(attempts.windows(2).all(|pair| pair[0].graph_identity_sha256
            == pair[1].graph_identity_sha256
            && pair[0].bounds_identity_sha256 == pair[1].bounds_identity_sha256
            && pair[0].input_identity_sha256 == pair[1].input_identity_sha256
            && pair[0].target_set_identity_sha256 != pair[1].target_set_identity_sha256));
    }

    #[test]
    fn phase_resident_refuses_oversized_dense_target_before_clone_prepare_or_backend() {
        let (graph, mut bounds, input) = phase_resident_fixture(16, 4);
        let before = map_bits(&bounds);
        let backend = DeclineUntilSmallBackend::new(20).with_preferred_rows(16);
        let deadline = Instant::now() + Duration::from_secs(5);

        assert!(
            freeze_phase_resident_targets(&graph, &input, &bounds, deadline, 16, 16, 4, 3,)
                .is_none()
        );

        assert_eq!(
            root_phase_resident_gpu_intermediate_sweep(
                &graph,
                &input,
                &backend,
                deadline,
                16,
                16,
                32,
                4,
                3,
                8 * 1024 * 1024 * 1024,
                &mut bounds,
            ),
            RootIntermediateSweepAttempt::CleanDecline
        );
        assert!(backend.attempts.lock().expect("attempt lock").is_empty());
        assert_eq!(map_bits(&bounds), before);

        // Pin the admission ordering that makes the runtime refusal above a
        // no-deep-clone path: impossible target count precedes both reserves,
        // and the complete checked borrowed-row census precedes the clone loop.
        let source = include_str!("intermediate_sweep.rs");
        let body = source
            .split_once("fn freeze_phase_resident_targets(")
            .expect("freeze helper declaration")
            .1
            .split_once("pub(in crate::beta_crown::engine::graph) fn root_phase_resident")
            .expect("resident route follows freeze helper")
            .0;
        let count_guard = body
            .find("dense_names.len() > max_dense_rows")
            .expect("target-count cap");
        let first_reserve = body.find("dense_set.try_reserve").expect("first reserve");
        let checked_rows = body
            .find("dense_rows = dense_rows.checked_add(live_bound.len())?")
            .expect("borrowed checked row census");
        let clone_loop = body
            .find("for (node_name, frozen_bound) in dense_borrowed")
            .expect("post-census clone loop");
        assert!(count_guard < first_reserve);
        assert!(checked_rows < clone_loop);
    }

    #[test]
    fn phase_resident_target_identity_binds_the_target_role() {
        let (graph, bounds, input) = chain_fixture();
        let deadline = Instant::now() + Duration::from_secs(5);
        let prepare = |role| {
            prepare_root_intermediate_sweep(
                &graph,
                &input,
                &bounds,
                vec![SelectedSweepTarget {
                    node_name: "linear0".into(),
                    selected_rows: Arc::from(&[0_u32, 1][..]),
                    frozen_bound: bounds["linear0"].clone(),
                    role: Some(role),
                }],
                deadline,
            )
            .expect("role-bound target prepares")
        };
        let dense = prepare(RootIntermediateSweepTargetRole::DenseMandatory);
        let comprehensive = prepare(RootIntermediateSweepTargetRole::Comprehensive);
        assert_eq!(
            dense.plan.graph_identity_sha256,
            comprehensive.plan.graph_identity_sha256
        );
        assert_eq!(
            dense.plan.bounds_identity_sha256,
            comprehensive.plan.bounds_identity_sha256
        );
        assert_ne!(
            dense.plan.target_set_identity_sha256, comprehensive.plan.target_set_identity_sha256,
            "the role byte must be part of the echoed target-set authority"
        );
    }

    #[test]
    fn phase_resident_rejects_a_bad_late_target_without_dense_prefix_publication() {
        let (graph, mut bounds, input) = phase_resident_fixture(16, 4);
        let before = map_bits(&bounds);
        let backend = DeclineUntilSmallBackend::poison_last_target(20).with_preferred_rows(16);

        assert_eq!(
            root_phase_resident_gpu_intermediate_sweep(
                &graph,
                &input,
                &backend,
                Instant::now() + Duration::from_secs(5),
                16,
                16,
                32,
                4,
                512,
                8 * 1024 * 1024 * 1024,
                &mut bounds,
            ),
            RootIntermediateSweepAttempt::Failed
        );
        assert_eq!(backend.attempts.lock().expect("attempt lock").len(), 1);
        assert_eq!(map_bits(&bounds), before);
    }

    #[test]
    fn phase_resident_rejects_a_non_echoed_role_hash_before_publication() {
        let (graph, mut bounds, input) = phase_resident_fixture(16, 4);
        let before = map_bits(&bounds);
        let backend = DeclineUntilSmallBackend::new(20)
            .with_preferred_rows(16)
            .with_corrupt_target_identity();

        assert_eq!(
            root_phase_resident_gpu_intermediate_sweep(
                &graph,
                &input,
                &backend,
                Instant::now() + Duration::from_secs(5),
                16,
                16,
                32,
                4,
                512,
                8 * 1024 * 1024 * 1024,
                &mut bounds,
            ),
            RootIntermediateSweepAttempt::Failed
        );
        assert_eq!(backend.attempts.lock().expect("attempt lock").len(), 1);
        assert_eq!(map_bits(&bounds), before);
    }

    #[test]
    fn an_expired_entry_is_a_failure_and_cannot_authorize_legacy_fallback() {
        let backend = DeclineUntilSmallBackend::new(32);
        let graph = GraphNetwork::new();
        let input = bounded(&[-1.0], &[1.0]);
        let mut bounds = HashMap::new();
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("monotonic clock has a prior instant");

        assert_eq!(
            root_wide_demanded_intermediate_sweep(
                &graph,
                &input,
                &backend,
                expired,
                1,
                1,
                1,
                1,
                1,
                1,
                &mut bounds,
            ),
            RootIntermediateSweepAttempt::Failed
        );
        assert!(backend.attempts.lock().expect("attempt lock").is_empty());
    }

    #[test]
    fn clean_decline_retries_with_a_fresh_exact_transcript_and_accepts_once() {
        let dimension = 64usize;
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "stem",
            Layer::Reshape(ReshapeLayer::new(vec![dimension as i64])),
        ));
        graph.add_node(GraphNode::new(
            "stable_wide",
            Layer::Add(AddLayer),
            vec!["stem".into(), "stem".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["stable_wide".into()],
        ));
        graph.set_output("relu");
        graph
            .exec_order()
            .expect("cache deterministic execution order");

        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[dimension]), 0.5),
            ArrayD::from_elem(IxDyn(&[dimension]), 1.5),
        )
        .expect("input box");
        let mut bounds = HashMap::from([
            ("stem".into(), input.clone()),
            (
                "stable_wide".into(),
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[dimension]), 1.0),
                    ArrayD::from_elem(IxDyn(&[dimension]), 3.0),
                )
                .expect("stable target box"),
            ),
            (
                "relu".into(),
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[dimension]), 1.0),
                    ArrayD::from_elem(IxDyn(&[dimension]), 3.0),
                )
                .expect("relu box"),
            ),
        ]);
        let backend = DeclineUntilSmallBackend::new(32);

        assert_eq!(
            root_wide_demanded_intermediate_sweep(
                &graph,
                &input,
                &backend,
                Instant::now() + Duration::from_secs(5),
                dimension,
                dimension,
                64,
                1,
                1,
                512 * 1024 * 1024,
                &mut bounds,
            ),
            RootIntermediateSweepAttempt::Completed(1)
        );

        let attempts = backend.attempts.lock().expect("attempt lock");
        assert_eq!(
            attempts
                .iter()
                .map(|attempt| attempt.rows)
                .collect::<Vec<_>>(),
            [64, 32]
        );
        assert_eq!(*backend.accepted.lock().expect("accepted lock"), 1);
        assert_eq!(
            attempts[0].graph_identity_sha256,
            attempts[1].graph_identity_sha256
        );
        assert_eq!(
            attempts[0].bounds_identity_sha256,
            attempts[1].bounds_identity_sha256
        );
        assert_eq!(
            attempts[0].input_identity_sha256,
            attempts[1].input_identity_sha256
        );
        assert_ne!(
            attempts[0].target_set_identity_sha256, attempts[1].target_set_identity_sha256,
            "the reduced row set is a distinct exact request"
        );
        assert_eq!(
            bounds["stable_wide"]
                .lower()
                .iter()
                .filter(|&&value| value == 1.25)
                .count(),
            32
        );
        assert_eq!(
            bounds["stable_wide"]
                .upper()
                .iter()
                .filter(|&&value| value == 2.75)
                .count(),
            32
        );
    }

    #[test]
    fn multi_depth_chain_tape_emits_identity_and_exact_canonical_hashes() {
        let (graph, bounds, input) = chain_fixture();
        let deadline = Instant::now() + Duration::from_secs(5);
        let prepared = prepare_root_intermediate_sweep(
            &graph,
            &input,
            &bounds,
            vec![
                selected("linear0", &[1], &bounds),
                selected("linear1", &[0], &bounds),
            ],
            deadline,
        )
        .expect("multi-depth chain plan");
        prepared.plan.validate().expect("canonical plan");
        assert_eq!(prepared.plan.injections.len(), 2);
        assert_eq!(prepared.plan.total_rows, 2);
        assert!(prepared
            .plan
            .ops_backward
            .iter()
            .any(|op| matches!(op, GpuBackwardOp::Identity { .. })));
        assert!(prepared.plan.ops_backward.iter().all(|op| matches!(
            op,
            GpuBackwardOp::Unary { .. } | GpuBackwardOp::Identity { .. }
        )));
        assert!(prepared
            .plan
            .injections
            .windows(2)
            .all(|pair| (pair[0].slot, pair[0].target_id) < (pair[1].slot, pair[1].target_id)));

        let mut reversed = HashMap::new();
        for name in ["linear1", "relu0", "reshape", "linear0"] {
            reversed.insert(name.to_string(), bounds[name].clone());
        }
        let replay = prepare_root_intermediate_sweep(
            &graph,
            &input,
            &reversed,
            vec![
                selected("linear0", &[1], &reversed),
                selected("linear1", &[0], &reversed),
            ],
            deadline,
        )
        .expect("map-order-independent replay");
        assert_eq!(
            prepared.plan.graph_identity_sha256,
            replay.plan.graph_identity_sha256
        );
        assert_eq!(
            prepared.plan.bounds_identity_sha256,
            replay.plan.bounds_identity_sha256
        );
        assert_eq!(
            prepared.plan.target_set_identity_sha256,
            replay.plan.target_set_identity_sha256
        );
        assert_eq!(prepared.input_identity_sha256, replay.input_identity_sha256);

        let mut changed = bounds;
        changed.insert(
            "reshape".into(),
            bounded(&[-1.9, -2.0, -3.0], &[2.0, 2.0, 3.0]),
        );
        let changed = prepare_root_intermediate_sweep(
            &graph,
            &input,
            &changed,
            vec![
                selected("linear0", &[1], &changed),
                selected("linear1", &[0], &changed),
            ],
            deadline,
        )
        .expect("one-bit-distinct transcript");
        assert_ne!(
            prepared.plan.bounds_identity_sha256,
            changed.plan.bounds_identity_sha256
        );
        assert_ne!(
            prepared.plan.graph_identity_sha256, changed.plan.graph_identity_sha256,
            "the baked ReLU descriptor must bind its exact source box"
        );
        assert_eq!(
            prepared.plan.target_set_identity_sha256,
            changed.plan.target_set_identity_sha256
        );
    }

    #[test]
    fn residual_add_builds_a_valid_reverse_dag_tape() {
        let mut graph = GraphNetwork::new();
        graph.add_node(linear("left", NETWORK_INPUT, arr2(&[[1.25]]), arr1(&[0.1])));
        graph.add_node(linear(
            "right",
            NETWORK_INPUT,
            arr2(&[[-0.75]]),
            arr1(&[-0.2]),
        ));
        graph.add_node(GraphNode::new(
            "sum",
            Layer::Add(AddLayer),
            vec!["left".into(), "right".into()],
        ));
        graph.set_output("sum");
        let bounds = HashMap::from([
            ("left".into(), bounded(&[-3.0], &[3.0])),
            ("right".into(), bounded(&[-3.0], &[3.0])),
            ("sum".into(), bounded(&[-6.0], &[6.0])),
        ]);
        let input = bounded(&[-1.0], &[2.0]);
        let prepared = prepare_root_intermediate_sweep(
            &graph,
            &input,
            &bounds,
            vec![selected("sum", &[0], &bounds)],
            Instant::now() + Duration::from_secs(2),
        )
        .expect("residual DAG plan");
        prepared.plan.validate().expect("valid reverse DAG");
        assert!(matches!(
            prepared.plan.ops_backward.first(),
            Some(GpuBackwardOp::Add { .. })
        ));
        assert_eq!(
            prepared
                .plan
                .ops_backward
                .iter()
                .filter(|op| matches!(op, GpuBackwardOp::Unary { .. }))
                .count(),
            2
        );
    }

    fn frozen(name: &str, id: u64, base: BoundedTensor, rows: &[u32]) -> FrozenSweepTarget {
        FrozenSweepTarget {
            target_id: id,
            node_name: name.into(),
            target_shape: Arc::from(base.shape().to_vec()),
            selected_rows: Arc::from(rows),
            frozen_bound: base,
            role: None,
        }
    }

    fn role_frozen(
        name: &str,
        id: u64,
        base: BoundedTensor,
        rows: &[u32],
        role: RootIntermediateSweepTargetRole,
    ) -> FrozenSweepTarget {
        FrozenSweepTarget {
            target_id: id,
            node_name: name.into(),
            target_shape: Arc::from(base.shape().to_vec()),
            selected_rows: Arc::from(rows),
            frozen_bound: base,
            role: Some(role),
        }
    }

    fn result(
        id: u64,
        rows: &[u32],
        lower: &[f32],
        upper: &[f32],
    ) -> ny_core::GpuIntermediateTargetResult {
        ny_core::GpuIntermediateTargetResult {
            target_id: id,
            row_offset: 0,
            selected_rows: Arc::from(rows),
            lower_bounds: lower.to_vec(),
            upper_bounds: upper.to_vec(),
        }
    }

    fn map_bits(bounds: &HashMap<String, BoundedTensor>) -> Vec<(String, Vec<u32>, Vec<u32>)> {
        let mut bits: Vec<_> = bounds
            .iter()
            .map(|(name, bound)| {
                (
                    name.clone(),
                    bound.lower().iter().map(|value| value.to_bits()).collect(),
                    bound.upper().iter().map(|value| value.to_bits()).collect(),
                )
            })
            .collect();
        bits.sort_by(|a, b| a.0.cmp(&b.0));
        bits
    }

    #[test]
    fn batch_publication_rejects_a_late_disjoint_target_without_a_prefix() {
        let base_a = bounded(&[-1.0, -2.0], &[1.0, 2.0]);
        let base_b = bounded(&[-1.0, -2.0], &[1.0, 2.0]);
        let targets = [
            frozen("a", 1, base_a.clone(), &[0]),
            frozen("b", 2, base_b.clone(), &[0]),
        ];
        let results = [
            result(1, &[0], &[-0.5], &[0.5]),
            result(2, &[0], &[3.0], &[4.0]),
        ];
        let mut live = HashMap::from([("a".into(), base_a), ("b".into(), base_b)]);
        let before = map_bits(&live);
        assert!(publish_validated_batch(
            &mut live,
            &targets,
            &results,
            Instant::now() + Duration::from_secs(2),
            || true,
        )
        .is_none());
        assert_eq!(map_bits(&live), before);
    }

    #[test]
    fn role_bound_publication_preserves_equal_endpoint_bits_rowwise() {
        let base = bounded(&[-0.0, -2.0], &[2.0, 2.0]);
        let targets = [role_frozen(
            "target",
            7,
            base.clone(),
            &[0, 1],
            RootIntermediateSweepTargetRole::DenseMandatory,
        )];
        let results = [result(7, &[0, 1], &[0.0, -1.0], &[2.0, 2.0])];
        let mut live = HashMap::from([("target".into(), base)]);

        assert_eq!(
            publish_validated_role_bound_batch(
                &mut live,
                &targets,
                &results,
                Instant::now() + Duration::from_secs(2),
                || true,
            ),
            Some(1)
        );
        assert_eq!(
            live["target"].lower()[[0]].to_bits(),
            (-0.0_f32).to_bits(),
            "numeric equality must keep the frozen endpoint bits"
        );
        assert_eq!(live["target"].lower()[[1]], -1.0);
        assert_eq!(live["target"].upper()[[0]].to_bits(), 2.0_f32.to_bits());
        assert_eq!(live["target"].upper()[[1]].to_bits(), 2.0_f32.to_bits());
    }

    #[test]
    fn role_bound_publication_preserves_exact_l2_annotation_bits() {
        let l2 = L2Constraint::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.0, 0.5]).expect("center shape"),
            ArrayD::from_elem(IxDyn(&[]), -0.0),
            0,
            &[2],
        )
        .expect("valid exact-bit annotation");
        let base = bounded(&[-1.0, -2.0], &[1.0, 2.0]).with_l2_constraint(l2.clone());
        let targets = [role_frozen(
            "target",
            7,
            base.clone(),
            &[0],
            RootIntermediateSweepTargetRole::DenseMandatory,
        )];
        let results = [result(7, &[0], &[-0.25], &[0.25])];
        let mut live = HashMap::from([("target".into(), base)]);

        assert_eq!(
            publish_validated_role_bound_batch(
                &mut live,
                &targets,
                &results,
                Instant::now() + Duration::from_secs(2),
                || true,
            ),
            Some(1)
        );
        let kept = live["target"]
            .l2_constraint()
            .expect("role-bound publication keeps L2");
        assert!(l2_constraint_bits_match(Some(kept), Some(&l2)));
        assert_eq!(kept.center()[[0]].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(kept.radius()[[]].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn role_bound_publication_rejects_signed_zero_l2_drift() {
        let frozen_l2 = L2Constraint::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.0, 0.5]).expect("frozen center"),
            ArrayD::from_elem(IxDyn(&[]), -0.0),
            0,
            &[2],
        )
        .expect("frozen annotation");
        let live_l2 = L2Constraint::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.5]).expect("live center"),
            ArrayD::from_elem(IxDyn(&[]), 0.0),
            0,
            &[2],
        )
        .expect("live annotation");
        assert_eq!(frozen_l2, live_l2, "semantic equality hides bit drift");
        assert!(!l2_constraint_bits_match(Some(&frozen_l2), Some(&live_l2)));

        let frozen = bounded(&[-1.0, -2.0], &[1.0, 2.0]).with_l2_constraint(frozen_l2);
        let live_bound = bounded(&[-1.0, -2.0], &[1.0, 2.0]).with_l2_constraint(live_l2);
        let targets = [role_frozen(
            "target",
            7,
            frozen,
            &[0],
            RootIntermediateSweepTargetRole::DenseMandatory,
        )];
        let results = [result(7, &[0], &[-0.25], &[0.25])];
        let mut live = HashMap::from([("target".into(), live_bound)]);

        assert!(publish_validated_role_bound_batch(
            &mut live,
            &targets,
            &results,
            Instant::now() + Duration::from_secs(2),
            || true,
        )
        .is_none());
        assert_eq!(live["target"].lower()[[0]], -1.0);
        assert_eq!(live["target"].upper()[[0]], 1.0);
    }

    #[test]
    fn role_bound_publication_rejects_a_stale_late_target_without_a_prefix() {
        let base_a = bounded(&[-2.0, -2.0], &[2.0, 2.0]);
        let base_b = bounded(&[-3.0, -3.0], &[3.0, 3.0]);
        let targets = [
            role_frozen(
                "a",
                1,
                base_a.clone(),
                &[0],
                RootIntermediateSweepTargetRole::DenseMandatory,
            ),
            role_frozen(
                "b",
                2,
                base_b,
                &[1],
                RootIntermediateSweepTargetRole::Comprehensive,
            ),
        ];
        let results = [
            result(1, &[0], &[-0.5], &[0.5]),
            result(2, &[1], &[-0.75], &[0.75]),
        ];
        let late_b = bounded(&[-2.5, -3.0], &[3.0, 3.0]);
        let mut live = HashMap::from([("a".into(), base_a), ("b".into(), late_b)]);
        let before = map_bits(&live);

        assert!(publish_validated_role_bound_batch(
            &mut live,
            &targets,
            &results,
            Instant::now() + Duration::from_secs(2),
            || true,
        )
        .is_none());
        assert_eq!(map_bits(&live), before);
    }

    #[test]
    fn authority_loss_after_staging_leaves_the_live_map_bit_identical() {
        let base = bounded(&[-1.0, -2.0], &[1.0, 2.0]);
        let targets = [frozen("target", 7, base.clone(), &[0])];
        let results = [result(7, &[0], &[-0.25], &[0.25])];
        let mut live = HashMap::from([("target".into(), base)]);
        let before = map_bits(&live);
        let authority_checks = std::cell::Cell::new(0usize);
        assert!(publish_validated_batch(
            &mut live,
            &targets,
            &results,
            Instant::now() + Duration::from_secs(2),
            || {
                authority_checks.set(authority_checks.get() + 1);
                false
            },
        )
        .is_none());
        assert_eq!(authority_checks.get(), 1);
        assert_eq!(map_bits(&live), before);
    }

    #[test]
    fn strict_endpoint_shrink_preserves_the_exact_live_l2_annotation() {
        let l2 = L2Constraint::new(
            ArrayD::from_elem(IxDyn(&[2]), 0.0),
            ArrayD::from_elem(IxDyn(&[]), 2.0),
            0,
            &[2],
        )
        .expect("valid one-slice constraint");
        let base = bounded(&[-1.0, -2.0], &[1.0, 2.0]).with_l2_constraint(l2.clone());
        let targets = [frozen("target", 7, base.clone(), &[0])];
        let results = [result(7, &[0], &[-0.25], &[0.25])];
        let mut live = HashMap::from([("target".into(), base)]);

        assert_eq!(
            publish_validated_batch(
                &mut live,
                &targets,
                &results,
                Instant::now() + Duration::from_secs(2),
                || true,
            ),
            Some(1)
        );
        assert_eq!(live["target"].lower()[[0]], -0.25);
        assert_eq!(live["target"].upper()[[0]], 0.25);
        assert_eq!(live["target"].l2_constraint(), Some(&l2));
    }

    #[test]
    fn valid_batch_commits_every_shrink_against_the_current_live_map() {
        let frozen_a = bounded(&[-2.0, -2.0], &[2.0, 2.0]);
        let frozen_b = bounded(&[-3.0, -3.0], &[3.0, 3.0]);
        let targets = [
            frozen("a", 1, frozen_a, &[0]),
            frozen("b", 2, frozen_b, &[1]),
        ];
        let results = [
            result(1, &[0], &[-0.5], &[0.5]),
            result(2, &[1], &[-0.75], &[0.75]),
        ];
        let mut live = HashMap::from([
            ("a".into(), bounded(&[-1.0, -1.0], &[1.0, 1.0])),
            ("b".into(), bounded(&[-1.0, -2.0], &[1.0, 2.0])),
        ]);
        assert_eq!(
            publish_validated_batch(
                &mut live,
                &targets,
                &results,
                Instant::now() + Duration::from_secs(2),
                || true,
            ),
            Some(2)
        );
        assert_eq!(live["a"].lower()[[0]], -0.5);
        assert_eq!(live["a"].upper()[[0]], 0.5);
        assert_eq!(live["b"].lower()[[1]], -0.75);
        assert_eq!(live["b"].upper()[[1]], 0.75);
    }
}
