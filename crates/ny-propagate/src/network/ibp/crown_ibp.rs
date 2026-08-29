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
use crate::network::crown_quality::{crown_gain_stat, CrownGainStat};
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
use tracing::{debug, info, warn};

/// #cgan-reclaim item 3 (docs/CGAN_BUDGET_RECLAIM_2026-08-20.md):
/// commit-what-you-have at a mid-walk GLOBAL deadline expiry, instead of the
/// historical `Err(DeadlineExceeded)` that discards EVERY completed per-node
/// tightening of this collection wholesale ("deadline-clipped cascade
/// tightenings thrown away", the live #18 signature on the scored cgan route,
/// which reaches this collector through the graph cascade's sequential
/// linear-chain fast path).
///
/// SOUNDNESS: every retained entry was independently certified before expiry —
/// each completed CROWN result was already shrink-only intersected against its
/// incumbent IBP bound at its own publication site (the same monotone
/// two-enclosures argument as `census_commit::shrink_only_commit`), and each
/// filled remainder is the sound IBP forward bound the historical per-node
/// fallback publishes for exactly this reason. The fill is marked
/// `ForwardFallback(DeadlineExceeded)` per layer, so completeness predicates
/// (`crown_ibp_result_is_complete`, `all_requested_crown_targets_completed`)
/// see the truncation and the collection cache stores it as truncated, never
/// serving it as complete. No new wall-clock authority: the only post-expiry
/// work is cloning the already-collected IBP tensors (bounded, front-loaded —
/// the same accepted residue class as `NY_PATCHES_FINITE_EXPIRY`).
fn commit_partial_on_expiry(
    network: &Network,
    ibp_bounds: &[BoundedTensor],
    mut crown_ibp_bounds: Vec<BoundedTensor>,
    mut provenance: Vec<BoundsProvenance>,
    mut fallback_events: Vec<CrownIbpFallbackEvent>,
    detail: &str,
) -> CrownIbpBoundsResult {
    let completed = crown_ibp_bounds.len();
    let total = ibp_bounds.len();
    for (k, ibp_bound) in ibp_bounds.iter().enumerate().skip(completed) {
        crown_ibp_bounds.push(ibp_bound.clone());
        provenance.push(BoundsProvenance::ForwardFallback(
            CrownIbpFallbackReason::DeadlineExceeded,
        ));
        fallback_events.push(CrownIbpFallbackEvent {
            layer_index: k,
            layer_type: network
                .layers
                .get(k)
                .map_or_else(|| "unknown".to_string(), |l| l.layer_type().to_string()),
            reason: CrownIbpFallbackReason::DeadlineExceeded,
            details: format!("#cgan-reclaim commit-what-you-have: {detail}"),
        });
    }
    eprintln!(
        "[cgan-reclaim] crown-ibp commit-what-you-have: tightened_layers={completed} \
         ibp_filled={} total={total} ({detail})",
        total - completed,
    );
    CrownIbpBoundsResult {
        bounds: crown_ibp_bounds,
        provenance,
        fallback_events,
    }
}

/// Classify a completed raw CROWN box before its sound intersection with IBP.
///
/// Guard-off avoids even the quality scan and returns the historical Crown
/// provenance exactly. Guard-on shares the graph-native integer-count
/// classifier and withholds Crown only for an almost-entirely non-finite result.
fn completed_crown_provenance(
    ibp: &BoundedTensor,
    crown: &BoundedTensor,
    honest_provenance_enabled: bool,
) -> (BoundsProvenance, Option<CrownGainStat>) {
    if !honest_provenance_enabled {
        return (BoundsProvenance::Crown, None);
    }
    let gain = crown_gain_stat(ibp, crown);
    let provenance = if gain.is_vacuous() {
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::CrownVacuousResult)
    } else {
        BoundsProvenance::Crown
    };
    (provenance, Some(gain))
}

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
    honest_provenance_enabled: bool,
    patches_deadline_flat_bias_enabled: Option<bool>,
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
    // COST-WEIGHTED sequential grants (#seq-weighted-budget, 2026-08-21).
    // Equal-share grants starved early heavy candidates: on
    // `cifar_bias_field_46` @ 60 s the layer-6 walk (8192 objective rows,
    // 7-layer walk) was granted 6.39 s against a 6.53 s need and degraded to
    // IBP — poisoning every later bound (output min_lo -37 → -1.4e7) — while
    // the collection finished with >30 s unused, because cheap wide layers
    // (16384 rows, 1-conv walks, 1.2 s spent) received the same share and
    // banked their surplus too late. Weight = objective rows × walk length
    // (`len × (k+1)`): both factors the walk cost actually scales with. The
    // shares go through the SAME weighted policy (cap + floor) the graph-α
    // lane uses. `NY_SEQ_WEIGHTED_BUDGET=0` restores count-based shares
    // byte-for-byte.
    // Declared lever (#seq-weighted-budget): the raw token is read once through
    // the registry so the receipt records it; the OPT-OUT predicate is unchanged
    // (`!= "0"`) and now lives with the gate's cell in `preset_gates`.
    //
    // DELIVERY SEAM (#lever-reachability): `NY_SEQ_WEIGHTED_BUDGET` still wins
    // in BOTH directions where present; absent, the typed
    // `bab.seq_weighted_budget` key decides; absent from both, the shipped
    // ARMED behaviour stands. A candidate that loses its grant degrades to IBP
    // and that degraded intermediate is published into every downstream bound,
    // so the count-based control arm needed a route a scored run can take.
    let seq_weighted_budget = ny_levers::gate::resolve_raw_predicate(
        &ny_levers::decls::seq_weighted_budget::SEQ_WEIGHTED_BUDGET,
        crate::preset_gates::seq_weighted_budget_cell(),
        // The shipped state is ARMED (engaged by ABSENCE), which is exactly
        // why the cell must be tri-state: `unwrap_or_default()` here would
        // silently disarm the lane on every run.
        true,
        |raw| raw != Some("0"),
    );
    let global_budget_candidate_weights: Vec<f64> = ibp_bounds
        .iter()
        .enumerate()
        .map(|(layer_index, ibp_bound)| {
            if global_budget_candidate_mask[layer_index] {
                (ibp_bound.len().max(1) as f64) * ((layer_index + 1) as f64)
            } else {
                0.0
            }
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
        // #cgan-reclaim (default dark ⇒ byte-identical): armed, retain the
        // completed per-node tightenings instead — see `commit_partial_on_expiry`.
        if deadline.is_some_and(|d| Instant::now() >= d) {
            if crate::cgan_reclaim::cgan_reclaim_armed() {
                return Ok(commit_partial_on_expiry(
                    network,
                    &ibp_bounds,
                    crown_ibp_bounds,
                    provenance,
                    fallback_events,
                    &format!("global deadline expired before layer {k}/{n}"),
                ));
            }
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

        let mut granted_budget_secs = f64::INFINITY;
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
                let per_node_secs = if seq_weighted_budget {
                    let remaining_weight_sum: f64 = global_budget_candidate_weights[k..]
                        .iter()
                        .copied()
                        .sum();
                    crate::network::graph_alpha::budget_policy::compute_weighted_per_node_budget_secs(
                        remaining_secs,
                        remaining_weight_sum,
                        global_budget_candidate_weights[k],
                        ibp_bound.len().max(1) as f64,
                        per_node_time_budget,
                    )
                    .unwrap_or(remaining_secs)
                } else {
                    compute_global_per_node_budget_secs(
                        remaining_secs,
                        remaining_candidates,
                        per_node_time_budget,
                    )
                    .unwrap_or(remaining_secs)
                };
                granted_budget_secs = per_node_secs;
                Some(now + Duration::from_secs_f64(per_node_secs))
            });
            if deadline.is_some() && per_node_deadline.is_none() {
                // #cgan-reclaim: same expiry salvage as the loop-top site.
                if crate::cgan_reclaim::cgan_reclaim_armed() {
                    return Ok(commit_partial_on_expiry(
                        network,
                        &ibp_bounds,
                        crown_ibp_bounds,
                        provenance,
                        fallback_events,
                        &format!("global deadline expired while budgeting layer {k}/{n}"),
                    ));
                }
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
                    patches_deadline_flat_bias_enabled,
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
                            // Measure the raw CROWN box before intersection.
                            // The graph-native collector uses this same shared
                            // classifier; threading the collector-entry snapshot
                            // here keeps linear-chain fast paths under identical
                            // provenance/cache policy.
                            let (quality_provenance, gain) = completed_crown_provenance(
                                ibp_bound,
                                &cb,
                                honest_provenance_enabled,
                            );
                            match ibp_bound.intersection_per_element(&cb) {
                                // Per-element intersection succeeded (#2935).
                                Some((tightened, disjoint)) => {
                                    if disjoint > 0 {
                                        debug!(
                                            "CROWN-IBP layer {} ({}): per-element intersection: {} of {} elements disjoint, used union fallback",
                                            k, network.layers[k].layer_type(), disjoint, tightened.len()
                                        );
                                    }
                                    if let BoundsProvenance::ForwardFallback(reason) =
                                        quality_provenance
                                    {
                                        let stat = gain.expect(
                                            "honest fallback classification always records gain",
                                        );
                                        warn!(
                                            "CROWN-IBP layer {} ({}): CROWN result is VACUOUS -- \
                                             {}/{} finite IBP elements lost, {}/{} improvable \
                                             elements tightened materially; recording {:?}, not Crown",
                                            k,
                                            network.layers[k].layer_type(),
                                            stat.crown_lost,
                                            stat.ibp_finite,
                                            stat.tightened_material,
                                            stat.improvable,
                                            reason,
                                        );
                                        (
                                            tightened,
                                            BoundsProvenance::ForwardFallback(reason),
                                            Some((
                                                reason,
                                                format!(
                                                    "CROWN lost finiteness on {}/{} elements, \
                                                     tightened {}/{} improvable materially \
                                                     ({} disjoint)",
                                                    stat.crown_lost,
                                                    stat.ibp_finite,
                                                    stat.tightened_material,
                                                    stat.improvable,
                                                    disjoint,
                                                ),
                                            )),
                                        )
                                    } else {
                                        if let Some(stat) = gain.filter(|stat| stat.is_regressed())
                                        {
                                            info!(
                                                "CROWN-IBP layer {} ({}): CROWN result REGRESSED -- \
                                                 materially wider than IBP on {}/{} improvable \
                                                 elements while tightening {}",
                                                k,
                                                network.layers[k].layer_type(),
                                                stat.widened_material,
                                                stat.improvable,
                                                stat.tightened_material,
                                            );
                                        }
                                        (tightened, quality_provenance, None)
                                    }
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
                            // #cgan-reclaim: same expiry salvage as the loop-top
                            // site. Layer k's walk was cut, so k joins the IBP
                            // fill; layers 0..k keep their certified tightenings.
                            if crate::cgan_reclaim::cgan_reclaim_armed() {
                                return Ok(commit_partial_on_expiry(
                                    network,
                                    &ibp_bounds,
                                    crown_ibp_bounds,
                                    provenance,
                                    fallback_events,
                                    &format!(
                                        "global deadline expired during layer {k}/{n} walk: {msg}"
                                    ),
                                ));
                            }
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
            // #cgan-reclaim: layer k's result completed LATE, so it is dropped
            // (the established late-publication rule, `publish_concretized_crown`);
            // layers 0..k keep their certified tightenings, k.. fill from IBP.
            if crate::cgan_reclaim::cgan_reclaim_armed() {
                return Ok(commit_partial_on_expiry(
                    network,
                    &ibp_bounds,
                    crown_ibp_bounds,
                    provenance,
                    fallback_events,
                    &format!("global deadline expired before publishing layer {k}/{n}"),
                ));
            }
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
        if needs_crown
            && ny_levers::read(&ny_levers::decls::diagnostics::DUMP_NODE_BOUNDS)
                .value
                .as_bool()
        {
            // #budget-log: granted-vs-spent per collection walk, so a per-node
            // degrade (the layer-6 poisoning of 2026-08-21) is attributable to
            // the specific slice that missed rather than inferred from dumps.
            eprintln!(
                "[budget-log] layer={} type={} granted={:.3}s spent={:.3}s outcome={}",
                k,
                network.layers[k].layer_type(),
                granted_budget_secs,
                node_secs,
                match &provenance_tag {
                    BoundsProvenance::Crown => "crown".to_string(),
                    BoundsProvenance::ForwardFallback(reason) => format!("fallback:{reason:?}"),
                },
            );
        }
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
        // #cgan-reclaim: every layer completed and was published IN TIME (each
        // passed its own pre-publication expiry check); only this final
        // bookkeeping instant is late. Armed, return the fully-computed sound
        // result instead of discarding all n certified tightenings.
        if crate::cgan_reclaim::cgan_reclaim_armed() {
            return Ok(commit_partial_on_expiry(
                network,
                &ibp_bounds,
                crown_ibp_bounds,
                provenance,
                fallback_events,
                "global deadline expired before collection publication",
            ));
        }
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
            "[group-split] engine_attempt={:.3}s fill={:.3}s gemm={:.3}s engine_hits={} engine_misses={}",
            crate::layers::convolution::conv2d::ops_transpose_gemm::CONV_GROUP_ENGINE_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,
            crate::layers::convolution::conv2d::ops_transpose_gemm::CONV_GROUP_FILL_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,
            crate::layers::convolution::conv2d::ops_transpose_gemm::CONV_GROUP_GEMM_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,
            crate::layers::convolution::conv2d::ops_transpose_gemm::CONV_GROUP_ENGINE_HITS
                .load(std::sync::atomic::Ordering::Relaxed),
            crate::layers::convolution::conv2d::ops_transpose_gemm::CONV_GROUP_ENGINE_MISSES
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        eprintln!(
            "[engine-split] fill={:.3}s gemm={:.3}s scan={:.3}s macs={} tile_mm={:.3}s tile_copy={:.3}s",
            crate::layers::convolution::conv2d::ops_transpose_gemm::CONV_GROUP_ENGINE_FILL_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,
            crate::layers::convolution::conv2d::ops_transpose_gemm::CONV_GROUP_ENGINE_GEMM_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,
            crate::layers::convolution::conv2d::ops_transpose_gemm::CONV_GROUP_ENGINE_SCAN_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,
            crate::layers::convolution::conv2d::ops_transpose_gemm::CONV_GROUP_ENGINE_MACS
                .load(std::sync::atomic::Ordering::Relaxed),
            crate::faer_parallelism::FAER_TILE_MATMUL_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,
            crate::faer_parallelism::FAER_TILE_COPY_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,
        );
        eprintln!(
            "[walk-share] conv={:.3}s act={:.3}s other={:.3}s final={:.3}s act_bwd={:.3}s act_fold={:.3}s",
            super::crown_partial::WALK_CONV_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e9,
            super::crown_partial::WALK_ACT_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e9,
            super::crown_partial::WALK_OTHER_NANOS.load(std::sync::atomic::Ordering::Relaxed)
                as f64
                / 1e9,
            super::crown_partial::WALK_FINAL_NANOS.load(std::sync::atomic::Ordering::Relaxed)
                as f64
                / 1e9,
            crate::network::core::sequential::crown::backward_step::ACT_BWD_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e9,
            crate::network::core::sequential::crown::backward_step::ACT_FOLD_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e9,
        );
        eprintln!(
            "[err-split] scan={:.3}s pub={:.3}s",
            crate::layers::convolution::crown_helpers::CONV_ERR_SCAN_NANOS
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e9,
            crate::layers::convolution::crown_helpers::CONV_ERR_PUB_NANOS
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

#[cfg(test)]
mod honest_provenance_tests {
    use super::completed_crown_provenance;
    use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    fn bounds(lower: f32, upper: f32) -> BoundedTensor {
        BoundedTensor::new_allow_infinite(
            ArrayD::from_elem(IxDyn(&[100]), lower),
            ArrayD::from_elem(IxDyn(&[100]), upper),
        )
        .expect("valid test bounds")
    }

    #[test]
    fn sequential_publication_honors_the_shared_vacuity_guard() {
        let ibp = bounds(-1.0, 1.0);
        let vacuous = bounds(f32::NEG_INFINITY, f32::INFINITY);

        let (guarded, gain) = completed_crown_provenance(&ibp, &vacuous, true);
        assert_eq!(
            guarded,
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::CrownVacuousResult)
        );
        assert!(gain.is_some_and(|stat| stat.is_vacuous()));

        let (historical, gain) = completed_crown_provenance(&ibp, &vacuous, false);
        assert_eq!(historical, BoundsProvenance::Crown);
        assert_eq!(gain, None, "guard-off must skip the quality scan");

        let (healthy, gain) = completed_crown_provenance(&ibp, &bounds(-0.5, 0.5), true);
        assert_eq!(healthy, BoundsProvenance::Crown);
        assert!(gain.is_some_and(|stat| !stat.is_vacuous()));
    }
}
