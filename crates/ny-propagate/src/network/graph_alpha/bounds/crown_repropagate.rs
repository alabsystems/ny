// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#crown-repropagate`: push CROWN intermediate tightening FORWARD so it compounds.
//!
//! # The defect
//!
//! In `collect_crown_ibp_bounds_core_inner_with_cut_segment`, `ibp_bounds` is the
//! one-shot forward IBP map computed BEFORE the per-node tightening loop and it is
//! never written inside it (every reference is a `get`). Tightened results land in
//! a separate `crown_ibp_bounds` map, and every skipped/degraded node stores
//! `ibp_bound.clone()` — the PRE-tightening value. So a CROWN result at node `v` is
//! discarded for everything downstream of `v` that is not itself demanded.
//!
//! Diagnosed in `daebf092` / `docs/YOLO_DEMAND_SKIP_DISCARDS_TIGHTENING_2026-07-29.md`:
//! on TinyYOLO, `Add_8` is CROWN-tightened to width 0.6461 while its DIRECT consumer
//! `Relu_9` reports 8.2257 — impossible for a monotone 1-Lipschitz ReLU unless its
//! bound descends from an untightened `Add_8`. 12 of 15 nodes were skipped.
//!
//! # The sweep
//!
//! After the loop, walk `exec_order` (topological) once and re-apply each node's OWN
//! sound IBP transfer to its already-tightened inputs, intersecting the result back
//! in. One pass suffices: in topological order every predecessor is finalized before
//! its dependents, so the tightening compounds along the whole chain.
//!
//! # Why this is sound
//!
//! Both operands of the intersection are valid enclosures of the SAME node over the
//! SAME input box (`crown_ibp_bounds` only ever holds valid enclosures, and every
//! entry in it is keyed by node name for this one collection's `input`), so their
//! intersection is a valid enclosure.
//!
//! Three separate guards make that argument robust rather than merely plausible:
//!
//! 1. **Certified transfers only.** This sweep INTERSECTS, so a recomputed bound even
//!    1 ULP too tight would shave the stored sound bound and could exclude the true
//!    pre-activation — the exact false-verified vector `ibp.rs` documents for the
//!    plain conv forward. A blanket `_ => layer.propagate_ibp(..)` is therefore NOT
//!    admissible. Every admitted op is either EXACT in f32, routed to its certified
//!    `propagate_ibp_sound*` variant, or (binary arithmetic) given the same 1-ULP
//!    outward widening the sound graph forward gives it. Everything else is refused.
//! 2. **Strictly non-widening.** `intersection_per_element` UNIONS any disjoint
//!    element, so the result is adopted ONLY when `disjoint == 0`. The sweep can
//!    narrow an already-sound interval and do nothing else.
//! 3. **Crossed bounds fail closed.** A recomputed element with `lower > upper`
//!    (possible only under a bug or a NaN) necessarily makes `max(l) > min(u)` at
//!    that element, which counts as disjoint and therefore refuses the whole node.
//!    A crossed bound can certify anything; it is a −150 generator. Guard 2 is
//!    exactly that fail-closed check, and `intersection_per_element` independently
//!    returns `None` on any NaN endpoint.
//!
//! Any refusal (unadmitted op, missing input, transfer error, shape mismatch, NaN,
//! disjoint) leaves the node at its existing bound. Provenance is deliberately NOT
//! rewritten: these nodes did not run CROWN, and the honest-provenance counters must
//! keep saying so.
//!
//! Note that soundness here does NOT rest on the transfer being monotone in its input
//! bounds. Monotonicity decides whether the recomputed box is TIGHTER; guard 2 makes
//! a non-monotone (wider) result a no-op rather than a regression. That is why
//! `MulBinary`/`Div` are admissible at all: interval arithmetic treats their two
//! operands as independent, which over-approximates the dependency and is sound.

use crate::layers::{BoundPropagation, Layer};
use crate::network::core::graph::NETWORK_INPUT;
use crate::network::core::GraphNetwork;
use ny_core::Result;
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::Instant;
use tracing::info;

/// Master gate. Default OFF ⇒ the collector is byte-identical to the pre-sweep
/// behavior. `NY_CROWN_REPROPAGATE=1` arms the sweep.
pub(super) fn enabled() -> bool {
    std::env::var("NY_CROWN_REPROPAGATE").ok().as_deref() == Some("1")
}

/// Sweep policy, resolved from the environment ONCE per collection by
/// [`Options::from_env`].
///
/// Passing this explicitly rather than re-reading the environment inside the sweep
/// mirrors `collect_crown_ibp_bounds_core_inner_with_cut_segment`: tests inject the
/// policy directly so they never mutate the process-global environment that cargo's
/// parallel test threads share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Options {
    /// Arm the BINARY arm (`#crown-repropagate-binary`). Armed with the master gate;
    /// `NY_CROWN_REPROPAGATE_BINARY=0` restores the unary-only sweep that landed in
    /// `a233e0d6` exactly, which is what the A/B isolates the binary increment
    /// against.
    pub(super) binary_arm: bool,
    /// Per-node before/after width trace (`NY_CROWN_REPROPAGATE_DEBUG=1`).
    /// Diagnostic only — never feeds a verdict.
    pub(super) debug: bool,
}

impl Options {
    pub(super) fn from_env() -> Self {
        Self {
            binary_arm: std::env::var("NY_CROWN_REPROPAGATE_BINARY").ok().as_deref() != Some("0"),
            debug: std::env::var_os("NY_CROWN_REPROPAGATE_DEBUG")
                .is_some_and(|v| v != "0" && !v.is_empty()),
        }
    }
}

/// Outcome counters for one sweep. Public for tests and the summary line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RepropagateStats {
    /// Nodes whose stored bound was replaced by a strictly-narrower intersection.
    pub(super) repropagated: usize,
    /// Nodes visited and rejected (unadmitted op, missing input, transfer error,
    /// shape mismatch, NaN, or a disjoint/crossed element).
    pub(super) refused: usize,
    /// Subset of `repropagated` that went through the binary arm.
    pub(super) binary_repropagated: usize,
}

/// Resolve a node input to the box the sweep should feed the transfer.
///
/// `NETWORK_INPUT` is the collection's own input box (exact, not an enclosure of
/// anything looser). Everything else must already be present in the map under
/// construction; a missing entry means the node was never bounded this collection
/// and the sweep declines rather than inventing a box.
fn resolve<'a>(
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

/// CERTIFIED UNARY TRANSFER ALLOWLIST.
///
/// Admitted, per the certification survey in `graph_ibp.rs:60-90`:
///   * Linear / Conv1d / Conv2d / ConvTranspose{1,2}d — the shared dispatch routes
///     these to their CERTIFIED sound variants (directed rounding + Higham term).
///   * AveragePool — routed explicitly to `propagate_ibp_sound`; its PLAIN forward
///     is an uncertified accumulator whose f64 residual can exceed 1 f32 ULP under
///     ≥2^29 cancellation.
///   * Ops EXACT in f32: ReLU/Clip/Abs/Sign/Floor/Ceil/Round/Trunc, MaxPool's exact
///     max, and the pure data movers (Pad/Reshape/Flatten/Transpose/Squeeze/
///     Unsqueeze/Slice/Tile). Exact ⇒ no ULP widening is owed.
///
/// Everything else is refused: accumulators (Softmax, LayerNorm, Reduce*, CumSum,
/// MatMul, …) whose rounding residual is not certified, and transcendentals that
/// rest on a faithful-libm ASSUMPTION rather than a proof. Refusing only forgoes a
/// gain.
fn unary_transfer(
    layer: &Layer,
    input_bound: &BoundedTensor,
    engine: Option<&dyn ny_core::GemmEngine>,
    deadline: Option<Instant>,
) -> Option<Result<BoundedTensor>> {
    match layer {
        Layer::Linear(_)
        | Layer::Conv1d(_)
        | Layer::Conv2d(_)
        | Layer::ConvTranspose1d(_)
        | Layer::ConvTranspose2d(_) => {
            Some(super::ibp::propagate_node_ibp_with_engine_and_deadline(
                layer,
                input_bound,
                engine,
                deadline,
            ))
        }
        Layer::AveragePool(pool) => Some(pool.propagate_ibp_sound(input_bound)),
        Layer::ReLU(_)
        | Layer::Clip(_)
        | Layer::Abs(_)
        | Layer::Sign(_)
        | Layer::Floor(_)
        | Layer::Ceil(_)
        | Layer::Round(_)
        | Layer::Trunc(_)
        | Layer::MaxPool2d(_)
        | Layer::Pad(_)
        | Layer::Reshape(_)
        | Layer::Flatten(_)
        | Layer::Transpose(_)
        | Layer::Squeeze(_)
        | Layer::Unsqueeze(_)
        | Layer::Slice(_)
        | Layer::Tile(_) => Some(layer.propagate_ibp(input_bound)),
        _ => None,
    }
}

/// True for binary layers whose TWO graph inputs are unambiguously the two DATA
/// operands and whose interval transfer is elementwise (`#crown-repropagate-binary`).
///
/// This is the whole reason the sweep needed extending: a residual conv graph joins
/// at `Add` and ml4acopf's power-flow frontier is `MulBinary`, so a unary-only sweep
/// stops dead at exactly the node the tightening most needs to cross. On TinyYOLO the
/// unary sweep leaves `Add_15` at 338.7292 while its sibling chain reaches 16.79.
///
/// DELIBERATELY EXCLUDED, and not for soundness reasons alone:
///   * `MatMul` / `BilinearCrown` — ACCUMULATORS. Their rounding residual across the
///     contraction is not certified (same open item as the sound forward's generic
///     arm, `#sound-ibp-generic-arm`); a 1-ULP widening does not cover them.
///   * `Concat` — its second graph input may be a folded constant list; arity here
///     does not imply two data operands.
///   * `CompareTensor` / `ScatterAdd` / `IndexAdd` / `ScatterNd` / `ExpandLikeLastAxis`
///     — index/shape semantics where a raw two-input read is not the operand pair
///     `classify_node_inputs` would resolve.
///   * `Atan2` — transcendental; faithful-libm ASSUMPTION, not a certificate.
fn is_admitted_binary(layer: &Layer) -> bool {
    matches!(
        layer,
        Layer::Add(_)
            | Layer::Sub(_)
            | Layer::MulBinary(_)
            | Layer::Div(_)
            | Layer::MinBinary(_)
            | Layer::MaxBinary(_)
    )
}

/// Outward ULP widening owed by an admitted binary transfer.
///
/// Add/Sub/Mul/Div form each endpoint with a SINGLE f32 operation rounded to
/// NEAREST, which can round INWARD by up to half an ULP — inadmissible for a map
/// this sweep then intersects into. One outward ULP covers a single nearest
/// rounding, which is exactly the widening `propagate_ibp_sound_impl` applies to
/// these same layers in the sound graph forward (`graph_ibp.rs:1028`). Min/Max are
/// exact selections and owe nothing, but they take the same ULP: widening only ever
/// loosens, and one code path is easier to keep honest than two.
const BINARY_SOUNDNESS_ULPS: u32 = 1;

/// Push the tightening forward and report what moved.
///
/// `bounds` is mutated in place. Returns the outcome counters.
pub(super) fn sweep(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    exec_order: &[String],
    engine: Option<&dyn ny_core::GemmEngine>,
    deadline: Option<Instant>,
    options: Options,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> RepropagateStats {
    let Options { binary_arm, debug } = options;
    let mut stats = RepropagateStats::default();

    for node_name in exec_order.iter() {
        let Some(node) = graph.nodes.get(node_name) else {
            continue;
        };
        // The node must already hold a bound for this collection; the sweep only
        // ever narrows what the loop produced, it never introduces a new entry.
        if !bounds.contains_key(node_name) {
            continue;
        }

        let (recomputed, via_binary) = match node.inputs.as_slice() {
            [input_name] => {
                let Some(input_bound) = resolve(input_name, input, bounds) else {
                    stats.refused += 1;
                    continue;
                };
                match unary_transfer(&node.layer, input_bound, engine, deadline) {
                    Some(Ok(bound)) => (bound, false),
                    Some(Err(_)) | None => {
                        stats.refused += 1;
                        continue;
                    }
                }
            }
            [name_a, name_b] if binary_arm && is_admitted_binary(&node.layer) => {
                let (Some(a), Some(b)) = (
                    resolve(name_a, input, bounds),
                    resolve(name_b, input, bounds),
                ) else {
                    stats.refused += 1;
                    continue;
                };
                match node.layer.propagate_ibp_binary(a, b) {
                    Ok(mut bound) => {
                        bound.round_for_soundness_n_ulps_inplace(BINARY_SOUNDNESS_ULPS);
                        (bound, true)
                    }
                    Err(_) => {
                        stats.refused += 1;
                        continue;
                    }
                }
            }
            _ => {
                stats.refused += 1;
                continue;
            }
        };

        let existing = bounds
            .get(node_name)
            .expect("membership checked immediately above");
        let before_width = debug.then(|| existing.max_width());
        // `disjoint == 0` is the fail-closed guard: it rejects NaN-free crossed
        // bounds AND any element where the two enclosures do not overlap, either of
        // which would otherwise widen (or, for a crossed recompute, corrupt) the
        // stored bound. `None` covers shape mismatch and NaN.
        match existing.intersection_per_element(&recomputed) {
            Some((tightened, 0)) => {
                if let Some(before) = before_width {
                    eprintln!(
                        "[NY_CROWN_REPROPAGATE_V1] node='{node_name}' op={} arity={} \
                         width_before={before:.6} width_after={:.6}",
                        node.layer.layer_type(),
                        if via_binary { "binary" } else { "unary" },
                        tightened.max_width(),
                    );
                }
                bounds.insert(node_name.clone(), tightened);
                stats.repropagated += 1;
                if via_binary {
                    stats.binary_repropagated += 1;
                }
            }
            _ => stats.refused += 1,
        }
    }

    info!(
        "CROWN-IBP DAG #crown-repropagate: {} nodes re-tightened forward ({} via the \
         binary arm, binary_arm={binary_arm}), {} refused",
        stats.repropagated, stats.binary_repropagated, stats.refused,
    );
    stats
}
