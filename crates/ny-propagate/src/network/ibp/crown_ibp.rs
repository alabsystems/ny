// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN-IBP hybrid collection loop for sequential networks.
//!
//! The forward tightening helper lives in [`super::crown_ibp_forward`].

use super::crown_ibp_forward::propagate_forward_tightened_bound;
use super::crown_partial::{propagate_crown_partial_with_engine, PartialCrownPropagationResult};
use super::helpers::{
    check_sequential_ibp_nan, is_all_relu_stable, layer_output_needs_partial_crown,
};
use crate::layers::Layer;
use crate::network::core::Network;
use crate::network::graph_alpha::budget_policy::{
    compute_global_per_node_budget_secs, count_remaining_budget_candidates,
};
use crate::types::{
    BoundsProvenance, CrownIbpBoundsResult, CrownIbpFallbackEvent, CrownIbpFallbackReason,
    CrownIbpPerNodeTimeBudget,
};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Core CROWN-IBP collection with optional pre-computed IBP bounds (#3397).
///
/// When `precomputed_ibp` is `Some`, skips the internal `collect_ibp_bounds`
/// call, saving the entire IBP forward pass (~59s for soundnessbench).
pub(super) fn collect_core(
    network: &Network,
    input: &BoundedTensor,
    precomputed_ibp: Option<Vec<BoundedTensor>>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    per_node_time_budget: &CrownIbpPerNodeTimeBudget,
) -> Result<CrownIbpBoundsResult> {
    let n = network.layers.len();
    if n == 0 {
        return Ok(CrownIbpBoundsResult {
            bounds: vec![],
            provenance: vec![],
            fallback_events: vec![],
        });
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        // This status-bearing collector cannot publish a borrowed/fresh IBP
        // substitute without scanning or cloning every layer and constructing
        // matching per-layer provenance. Preserve hard authority instead.
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP: deadline exceeded before forward-bound collection".to_string(),
        ));
    }
    if network.has_self_attention() {
        return Err(NyError::UnsupportedConfiguration(
            "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
        ));
    }

    let ibp_bounds = match precomputed_ibp {
        Some(bounds) => {
            if bounds.len() != n {
                return Err(NyError::InvalidSpec(format!(
                    "pre-computed IBP bounds have {} entries, expected {} (one per layer)",
                    bounds.len(),
                    n
                )));
            }
            debug!(
                "CROWN-IBP: using pre-computed IBP bounds ({} layers), skipping IBP forward pass",
                n
            );
            bounds
        }
        None => super::forward::collect_ibp_bounds_with_deadline(network, input, deadline)?,
    };
    let mut crown_ibp_bounds = Vec::with_capacity(n);
    let mut provenance = Vec::with_capacity(n);
    let mut fallback_events = Vec::new();
    let global_budget_candidate_mask: Vec<bool> = ibp_bounds
        .iter()
        .enumerate()
        .map(|(layer_index, ibp_bound)| {
            let needs_crown = layer_output_needs_partial_crown(&network.layers, layer_index);
            let relu_stable_successor = needs_crown
                && matches!(network.layers.get(layer_index + 1), Some(Layer::ReLU(_)))
                && is_all_relu_stable(ibp_bound);
            needs_crown && !relu_stable_successor
        })
        .collect();
    // Per-node timing for CROWN-IBP collection (#3599).
    let collection_start = Instant::now();
    let mut crown_node_count = 0usize;
    let mut shortcut_node_count = 0usize;
    let mut deadline_node_count = 0usize;
    let mut crown_total_secs = 0.0f64;
    let mut shortcut_total_secs = 0.0f64;

    for (k, ibp_bound) in ibp_bounds.iter().enumerate() {
        // Once the global authority expires, this status-bearing API cannot
        // clone the remaining owned IBP tensors and allocate their per-layer
        // provenance. Preserve the typed deadline instead (#3328).
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "CROWN-IBP: deadline exceeded before layer {k}/{n}"
            )));
        }

        let node_start = Instant::now();
        let needs_crown = layer_output_needs_partial_crown(&network.layers, k);

        // ReLU-stable skip (#3599): if the successor is ReLU and every neuron
        // at this layer's output is stable (lower >= 0 or upper <= 0), CROWN
        // tightening cannot improve on IBP because the ReLU relaxation is
        // already exact. Keep this narrow: other nonlinear successors still
        // need tighter pre-activation bounds on one-sign intervals.
        let relu_stable_successor = needs_crown
            && matches!(network.layers.get(k + 1), Some(Layer::ReLU(_)))
            && is_all_relu_stable(ibp_bound);
        if relu_stable_successor {
            shortcut_node_count += 1;
            let node_secs = node_start.elapsed().as_secs_f64();
            shortcut_total_secs += node_secs;
            debug!(
                "CROWN-IBP layer {k}/{n} ({}): ReLU-stable skip — {} elements, IBP bounds exact",
                network.layers[k].layer_type(),
                ibp_bound.len(),
            );
            check_sequential_ibp_nan(
                ibp_bound,
                "Sequential CROWN-IBP",
                k,
                network.layers[k].layer_type(),
            )?;
            crown_ibp_bounds.push(ibp_bound.clone());
            provenance.push(BoundsProvenance::Crown);
            continue;
        }

        let (tightened, provenance_tag, fallback) = if needs_crown {
            let per_node_deadline = deadline.and_then(|d| {
                let now = Instant::now();
                if now >= d {
                    return None;
                }
                let remaining_secs = d.duration_since(now).as_secs_f64();
                let remaining_candidates =
                    count_remaining_budget_candidates(&global_budget_candidate_mask, k);
                // Use per-node budget share when the floor is met; otherwise
                // fall back to full remaining time. Avoids preemptive IBP
                // fallback for small networks where backward is fast (#4413).
                // crown_partial's per-layer deadline handles actual timeouts.
                let per_node_secs = compute_global_per_node_budget_secs(
                    remaining_secs,
                    remaining_candidates,
                    per_node_time_budget,
                )
                .unwrap_or(remaining_secs);
                Some(now + Duration::from_secs_f64(per_node_secs))
            });
            if deadline.is_some() && per_node_deadline.is_none() {
                return Err(NyError::DeadlineExceeded(format!(
                    "CROWN-IBP: deadline expired while budgeting layer {k}/{n}"
                )));
            } else {
                let partial_layers = &network.layers[0..=k];
                let crown_bounds = propagate_crown_partial_with_engine(
                    &network.layers,
                    input,
                    partial_layers,
                    &crown_ibp_bounds,
                    engine,
                    per_node_deadline,
                );
                match crown_bounds {
                    Ok(PartialCrownPropagationResult::Crown(cb)) => {
                        let cb = *cb;
                        // #3300: Shape-tolerant intersection. Reshape CROWN bounds to
                        // match IBP shape when element counts match but shapes differ
                        // (e.g., Reshape([-1]) produces [N] vs FlattenLayer(0) [1, N]).
                        let cb = if cb.shape() != ibp_bound.shape() && cb.len() == ibp_bound.len() {
                            debug!(
                            "CROWN-IBP layer {} ({}): reshaping CROWN {:?} to match forward {:?}",
                            k,
                            network.layers[k].layer_type(),
                            cb.shape(),
                            ibp_bound.shape()
                        );
                            match cb.reshape(ibp_bound.shape()) {
                                Ok(reshaped) => reshaped,
                                Err(_) => cb,
                            }
                        } else {
                            cb
                        };
                        if cb.shape() == ibp_bound.shape() {
                            match ibp_bound.intersection_per_element(&cb) {
                                // Per-element intersection succeeded (#2935).
                                Some((tightened, disjoint)) => {
                                    if disjoint > 0 {
                                        debug!(
                                            "CROWN-IBP layer {} ({}): per-element intersection: {} of {} elements disjoint, used union fallback",
                                            k, network.layers[k].layer_type(), disjoint, tightened.len()
                                        );
                                    }
                                    (tightened, BoundsProvenance::Crown, None)
                                }
                                // NaN or shape mismatch — full IBP fallback.
                                None => {
                                    debug!(
                                        "CROWN-IBP layer {} ({}): forward/CROWN intersection failed (NaN); using forward bound",
                                        k,
                                        network.layers[k].layer_type()
                                    );
                                    let reason = CrownIbpFallbackReason::EmptyIntersection;
                                    (
                                        ibp_bound.clone(),
                                        BoundsProvenance::ForwardFallback(reason),
                                        Some((
                                            reason,
                                            format!(
                                                "forward/CROWN intersection failed (NaN) for shape {:?}",
                                                ibp_bound.shape()
                                            ),
                                        )),
                                    )
                                }
                            }
                        } else {
                            debug!(
                                "CROWN-IBP layer {} ({}): shape mismatch CROWN {:?} vs forward {:?}; using forward bound",
                                k,
                                network.layers[k].layer_type(),
                                cb.shape(),
                                ibp_bound.shape()
                            );
                            let reason = CrownIbpFallbackReason::ShapeMismatch;
                            (
                                ibp_bound.clone(),
                                BoundsProvenance::ForwardFallback(reason),
                                Some((
                                    reason,
                                    format!(
                                        "crown shape {:?} does not match forward shape {:?}",
                                        cb.shape(),
                                        ibp_bound.shape()
                                    ),
                                )),
                            )
                        }
                    }
                    Ok(PartialCrownPropagationResult::ForwardFallback(fallback)) => (
                        ibp_bound.clone(),
                        BoundsProvenance::ForwardFallback(fallback.reason),
                        Some((fallback.reason, fallback.details)),
                    ),
                    // #3131, #3499: UnsupportedOp/Configuration/NumericalInstability →
                    // IBP fallback is correct. Other errors must propagate (#3106).
                    Err(
                        NyError::UnsupportedOp(msg)
                        | NyError::UnsupportedConfiguration(msg)
                        | NyError::NumericalInstability(msg),
                    ) => {
                        debug!(
                            "CROWN-IBP layer {} ({}): unsupported op, using forward bound: {}",
                            k,
                            network.layers[k].layer_type(),
                            msg
                        );
                        let reason = CrownIbpFallbackReason::CrownPropagationError;
                        (
                            ibp_bound.clone(),
                            BoundsProvenance::ForwardFallback(reason),
                            Some((reason, msg)),
                        )
                    }
                    Err(NyError::DeadlineExceeded(msg)) => {
                        if deadline.is_some_and(|limit| Instant::now() >= limit) {
                            return Err(NyError::DeadlineExceeded(msg));
                        }
                        deadline_node_count += 1;
                        debug!(
                        "CROWN-IBP layer {} ({}): per-node deadline exceeded, using forward bound: {}",
                        k,
                        network.layers[k].layer_type(),
                        msg
                    );
                        let reason = CrownIbpFallbackReason::PerNodeDeadlineExceeded;
                        (
                            ibp_bound.clone(),
                            BoundsProvenance::ForwardFallback(reason),
                            Some((reason, msg)),
                        )
                    }
                    // #3813: ShapeMismatch triggers IBP fallback. RSPLITTER models
                    // change intermediate dimensions, causing shape mismatches in
                    // partial CROWN backward. IBP fallback is always sound.
                    Err(NyError::ShapeMismatch {
                        ref expected,
                        ref got,
                    }) => {
                        let msg = format!("shape mismatch: expected {:?}, got {:?}", expected, got);
                        debug!(
                            "CROWN-IBP layer {} ({}): {}, using forward bound",
                            k,
                            network.layers[k].layer_type(),
                            msg
                        );
                        let reason = CrownIbpFallbackReason::ShapeMismatch;
                        (
                            ibp_bound.clone(),
                            BoundsProvenance::ForwardFallback(reason),
                            Some((reason, msg)),
                        )
                    }
                    Err(err) => return Err(err),
                }
            }
        } else {
            debug!(
                    "CROWN-IBP layer {} ({}): skipping redundant partial backward, next layer {} uses exact backward",
                    k,
                    network.layers[k].layer_type(),
                    network.layers[k + 1].layer_type()
                );
            propagate_forward_tightened_bound(
                &network.layers[k],
                k,
                input,
                &crown_ibp_bounds,
                ibp_bound,
            )?
        };
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(format!(
                "CROWN-IBP: deadline exceeded before publishing layer {k}/{n}"
            )));
        }
        check_sequential_ibp_nan(
            &tightened,
            "Sequential CROWN-IBP",
            k,
            network.layers[k].layer_type(),
        )?;

        // Per-node timing (#3599): track CROWN vs shortcut time distribution.
        let node_secs = node_start.elapsed().as_secs_f64();
        if needs_crown {
            crown_node_count += 1;
            crown_total_secs += node_secs;
        } else {
            shortcut_node_count += 1;
            shortcut_total_secs += node_secs;
        }
        if node_secs > 0.5 {
            info!(
                "CROWN-IBP: layer {k}/{n} ({}) took {node_secs:.3}s [{}]",
                network.layers[k].layer_type(),
                if needs_crown { "crown" } else { "shortcut" },
            );
        }

        crown_ibp_bounds.push(tightened);
        provenance.push(provenance_tag);
        if let Some((reason, details)) = fallback {
            if ny_levers::read(&ny_levers::decls::diagnostics::DUMP_NODE_BOUNDS)
                .value
                .as_bool()
            {
                eprintln!(
                    "[fb-event] layer={k} type={} reason={reason:?} details={details:?}",
                    network.layers[k].layer_type()
                );
            }
            fallback_events.push(CrownIbpFallbackEvent {
                layer_index: k,
                layer_type: network.layers[k].layer_type().to_string(),
                reason,
                details,
            });
        }
    }

    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP: deadline exceeded before collection publication".to_string(),
        ));
    }

    // CROWN-IBP collection timing summary (#3599).
    let collection_secs = collection_start.elapsed().as_secs_f64();
    if collection_secs > 0.1 {
        info!(
            "CROWN-IBP collection: {collection_secs:.3}s total, \
             {crown_node_count} crown nodes ({crown_total_secs:.3}s), \
             {shortcut_node_count} shortcut nodes ({shortcut_total_secs:.3}s), \
             {deadline_node_count} deadline-skipped",
        );
    }

    // Temporary diagnostic (NY_DUMP_NODE_BOUNDS=1): per-layer bound summary at
    // publication, for divergence hunting between binaries.
    if ny_levers::read(&ny_levers::decls::diagnostics::DUMP_NODE_BOUNDS)
        .value
        .as_bool()
    {
        let probe_nanos = crate::layers::linear::crown_single::INCOMING_ERR_NANOS
            .load(std::sync::atomic::Ordering::Relaxed);
        let probe_calls = crate::layers::linear::crown_single::INCOMING_ERR_CALLS
            .load(std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "[err-share] incoming_error_product: {probe_calls} calls, {:.3}s total",
            probe_nanos as f64 / 1e9
        );
        eprintln!(
            "[conv-share] bias={:.3}s err={:.3}s group={:.3}s group-engine={:.3}s",
            crate::layers::convolution::crown_helpers::CONV_BIAS_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e9,
            crate::layers::convolution::crown_helpers::CONV_ERR_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e9,
            crate::layers::convolution::conv2d::ops_transpose_gemm::CONV_GROUP_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e9,
            // group-engine is a SUBSET of group: the share of the conv-group
            // arm spent waiting on the sound_f64_gemm engine attempt. Printed
            // next to its parent so the reader can see the fraction directly,
            // which is the whole question 61c628a76 posed.
            crate::layers::convolution::conv2d::ops_transpose_gemm::CONV_GROUP_ENGINE_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e9,
        );
        eprintln!(
            "[patches-share] crown_elementwise_backward_patches: {} calls, {:.3}s total, {} serial-row calls",
            crate::layers::common::crown_patches::PATCHES_BWD_CALLS
                .load(std::sync::atomic::Ordering::Relaxed),
            crate::layers::common::crown_patches::PATCHES_BWD_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e9,
            crate::layers::common::crown_patches::PATCHES_BWD_SERIAL_ROWS
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        for (idx, bt) in crown_ibp_bounds.iter().enumerate() {
            let flat = bt.flatten();
            let (mut lo, mut hi, mut w) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
            for (&l, &u) in flat.lower().iter().zip(flat.upper().iter()) {
                lo = lo.min(l);
                hi = hi.max(u);
                w += f64::from(u) - f64::from(l);
            }
            eprintln!(
                "[node-dump] layer={idx} len={} min_lo={lo:.4} max_hi={hi:.4} width_sum={w:.2}",
                flat.len()
            );
        }
    }
    Ok(CrownIbpBoundsResult {
        bounds: crown_ibp_bounds,
        provenance,
        fallback_events,
    })
}
