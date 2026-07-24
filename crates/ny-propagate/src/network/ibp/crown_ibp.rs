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
        None => super::forward::collect_ibp_bounds(network, input)?,
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
    // Track whether deadline was exceeded — once exceeded, use IBP for all
    // remaining layers (sound but looser). Same pattern as GraphNetwork DAG
    // in graph_alpha/bounds/crown.rs (#3328).
    let mut deadline_exceeded = false;

    // Per-node timing for CROWN-IBP collection (#3599).
    let collection_start = Instant::now();
    let mut crown_node_count = 0usize;
    let mut shortcut_node_count = 0usize;
    let mut deadline_node_count = 0usize;
    let mut crown_total_secs = 0.0f64;
    let mut shortcut_total_secs = 0.0f64;

    for (k, ibp_bound) in ibp_bounds.iter().enumerate() {
        // Deadline check before each CROWN partial pass (#3328).
        if !deadline_exceeded {
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    info!(
                        "CROWN-IBP: deadline exceeded at layer {}/{}, remaining layers use IBP",
                        k, n
                    );
                    deadline_exceeded = true;
                }
            }
        }

        if deadline_exceeded {
            // Use IBP bounds directly — sound but looser than CROWN-IBP.
            deadline_node_count += 1;
            check_sequential_ibp_nan(
                ibp_bound,
                "Sequential CROWN-IBP",
                k,
                network.layers[k].layer_type(),
            )?;
            crown_ibp_bounds.push(ibp_bound.clone());
            provenance.push(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::DeadlineExceeded,
            ));
            fallback_events.push(CrownIbpFallbackEvent {
                layer_index: k,
                layer_type: network.layers[k].layer_type().to_string(),
                reason: CrownIbpFallbackReason::DeadlineExceeded,
                details: "deadline exceeded, using IBP bounds".to_string(),
            });
            continue;
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
                // Deadline truly expired between top-of-loop check and here.
                deadline_node_count += 1;
                debug!(
                    "CROWN-IBP layer {k}/{n} ({}): deadline expired during budget computation, using forward bound",
                    network.layers[k].layer_type(),
                );
                (
                    ibp_bound.clone(),
                    BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                    ),
                    Some((
                        CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                        "deadline expired during budget computation".to_string(),
                    )),
                )
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
            fallback_events.push(CrownIbpFallbackEvent {
                layer_index: k,
                layer_type: network.layers[k].layer_type().to_string(),
                reason,
                details,
            });
        }
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

    Ok(CrownIbpBoundsResult {
        bounds: crown_ibp_bounds,
        provenance,
        fallback_events,
    })
}
