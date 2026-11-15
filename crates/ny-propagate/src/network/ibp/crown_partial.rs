// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Partial CROWN backward dispatch for CROWN-IBP tightening.
//!
//! Contains the CPU Patches/Dense backward loop. GPU fast-path is in
//! `crown_partial_gpu.rs`.

use super::crown_partial_gpu::try_gpu_crown_partial_backward;
use super::forward::propagate_ibp_certified;
use super::helpers::{memory_budget_partial_fallback, patches_dense_materialization_fallback};
use super::sparse_merge::{
    find_unstable_dense_indices, merge_sparse_crown_with_ibp, scatter_sparse_crown_into_ibp,
};
use crate::bounds::patches::{CrownBounds, PatchGeometry, PatchesLinearBounds};
use crate::bounds::LinearBounds;
use crate::contiguous_flat_slice;
use crate::layers::Layer;
use crate::network::core::{
    crown_backward_step_patches,
    crown_backward_step_patches_with_deadline_authority_and_flat_bias_policy,
    materialize_terminal_crown_bounds_with_deadline, CrownStepFallback, CrownStepResult,
};
use crate::network::crown_memory::{cpu_crown_dense_budget_bytes, DenseMaterializationEstimate};
use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::debug;

pub(super) enum PartialCrownPropagationResult {
    Crown(Box<BoundedTensor>),
    ForwardFallback(CrownStepFallback),
}

/// Publish a completed CPU CROWN result only while its node budget is live.
///
/// The backward loop polls the deadline between layers, but concretization and
/// sparse post-processing can themselves cross the cutoff. Treat that late
/// result exactly like any other per-node timeout so the collection caller
/// keeps the sound forward bound instead.
fn publish_concretized_crown(
    result: BoundedTensor,
    deadline: Option<Instant>,
    completed_at: Instant,
) -> Result<BoundedTensor> {
    if deadline.is_some_and(|limit| completed_at >= limit) {
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP partial: per-node deadline exceeded after concretization".to_string(),
        ));
    }
    Ok(result)
}

fn concretization_memory_fallback(error: &NyError) -> Option<CrownStepFallback> {
    matches!(error, NyError::CpuMemoryExceeded { .. }).then(|| CrownStepFallback {
        reason: crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded,
        details: format!("CROWN-IBP partial concretization exceeded its CPU budget: {error}"),
    })
}

fn check_partial_deadline(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(NyError::DeadlineExceeded(
            "CROWN-IBP partial: deadline exceeded during final reshape".into(),
        ))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// #crown-partial-pnc — patch-native TERMINAL concretization
// ---------------------------------------------------------------------------
//
// WHY. On challenging_certified_training_2026 `cifar10_eps2_cnn7` (idx9890,
// 30 s) the sequential CROWN-IBP root collection never finishes, so BaB never
// starts (`[bab-stop] status=timeout explored=0`). The conv targets end their
// backward walk as a 6D Patches carrier over the 3x32x32 input, and the walk
// then DENSIFIED that carrier (`materialize_terminal_crown_bounds_with_deadline`,
// 65536 x 3072 per side on target 0) only to concretize every row across ALL
// 3072 input columns: 201M terms per side for a row whose receptive field is
// 27 taps. Target 0 alone spent 0.449 s densifying + 2.587 s concretizing
// (Mac DIAG log; the GB10 control log shows the same target at 3.405 s).
//
// WHAT. When the terminal carrier is a full-grid Affine Patches relation (no
// `unstable_idx`, no sparse seed of either kind) whose dense pair is WITHIN the
// CPU dense budget (the over-budget case is refused to IBP by
// `patches_dense_materialization_fallback` before this point, unchanged),
// concretize it directly from the taps with
// `PatchesLinearBounds::concretize_sound_sparse` — the same call the
// graph-native collector already makes (`graph_alpha/bounds/target_backward.rs`
// `#patches-sparse-concretize`) — instead of densifying.
//
// SOUNDNESS / BIT-IDENTITY. `concretize_sound_sparse` is BIT-FOR-BIT equal to
// `to_dense()?.concretize_sound(input)` on every layout it accepts
// (`bounds/patches/sparse_concretize.rs` module docs; pinned by
// `bounds::patches::tests::sparse_concretize_matches_dense_bit_identical`,
// including carried `coeff_err` on one and both sides): both routes run the
// same certified per-row kernel `concretize_row_directed`, and every column the
// taps skip holds an exact-zero coefficient and exact-zero error, i.e. an f64
// no-op add. The dense route here is exactly that pair
// (`to_dense_with_deadline_for_purpose` differs from `to_dense` only in
// telemetry; `concretize_sound_with_deadline` is `concretize_sound`'s fallible
// core). The ONE documented deviation: a non-finite coefficient degrades only
// its own row to `[-inf, +inf]`, where the dense NaN firewall degrades every
// row — both sound. The published value is then reshaped, published through
// `publish_concretized_crown` and intersected with the certified IBP box by the
// collector, exactly as before.
//
// FAIL-CLOSED ROUTING. Only a SUCCESSFUL patch-native result replaces the
// historical path. `DeadlineExceeded` propagates (the dense route would hit the
// same absolute limit at its first checkpoint). EVERY other error —
// `UnsupportedOp` (7D explicit rows, anchored sparse, identity mismatch),
// `CpuMemoryExceeded`, shape or layout refusals — falls through to the
// historical densify-and-concretize path, which then decides exactly as it
// always did. So outside a successful engagement the published bound is
// byte-identical to the pre-change tree.
//
// OBSERVABLES (unconditional stderr; "demand the lever's own observable"):
//   * once per process, on the first successful engagement:
//     `[crown-partial-pnc] ENGAGED rows=… in_dim=… taps_max=… …`
//   * every partial walk slower than `PNC_WALK_REPORT_SECS`, on EVERY route:
//     `[crown-partial-pnc] walk=<target> backward_s=… dense_s=… concretize_s=…
//      route=<route>` — the per-phase split the GB10 log was missing, and the
//     route that decided each target (`patch-native`, `dense-unsupported`,
//     `dense-refused`, or the out-of-scope reason).
//
// No lever: this is a module constant, the task contract forbids new raw
// environment reads, and a bit-identical representation change has no value
// for a control arm to preserve. Flip the constant to `false` to restore the
// historical route everywhere.
const PATCH_NATIVE_TERMINAL_CONCRETIZE: bool = true;

/// A partial walk at least this slow prints its `[crown-partial-pnc] walk=`
/// line. Matches the historical `debug!` per-partial timing threshold.
const PNC_WALK_REPORT_SECS: f64 = 0.5;

/// Terminal published patch-natively (the in-scope success arm).
const ROUTE_PATCH_NATIVE: &str = "patch-native";
/// In scope, but the patch-native kernel typed-refused the layout; the
/// historical densify path decided.
const ROUTE_DENSE_UNSUPPORTED: &str = "dense-unsupported";
/// In scope, but the patch-native kernel refused for another non-deadline
/// reason (memory admission, shape/layout validation); the historical densify
/// path decided.
const ROUTE_DENSE_REFUSED: &str = "dense-refused";
/// Route disabled by `PATCH_NATIVE_TERMINAL_CONCRETIZE` (or a test arm).
const ROUTE_DENSE_DISABLED: &str = "dense-disabled";
/// Out of scope: the SPARSE Patches seed, whose tracked-row merge follows.
const ROUTE_SPARSE_SEED: &str = "sparse-seed";
/// Out of scope: the unstable-rows DENSE seed, whose scatter follows.
const ROUTE_DENSE_UNSTABLE_ROWS: &str = "dense-unstable-rows";
/// Out of scope: the terminal carrier is already Dense.
const ROUTE_DENSE_CARRIER: &str = "dense-carrier";
/// Out of scope: the terminal Patches carrier holds an `unstable_idx` layout.
const ROUTE_SPARSE_LAYOUT: &str = "sparse-layout";
/// Out of scope: an Anchored (ConvTranspose-composed) window geometry. The
/// kernel accepts it, but its dense-parity oracle pins only Affine layouts, so
/// this route stays on what is pinned.
const ROUTE_ANCHORED_GEOMETRY: &str = "anchored-geometry";

/// Pure scope predicate for the patch-native terminal route. `Ok` hands back
/// the in-scope Patches carrier; `Err` names the out-of-scope route label.
///
/// Called only AFTER `patches_dense_materialization_fallback` admitted the
/// terminal, so an in-scope carrier's dense pair is within
/// `cpu_crown_dense_budget_bytes()`.
fn patch_native_terminal_scope(
    enabled: bool,
    patches_sparse_seed: bool,
    dense_unstable_rows_seed: bool,
    crown_bounds: &CrownBounds,
) -> std::result::Result<&PatchesLinearBounds, &'static str> {
    if !enabled {
        return Err(ROUTE_DENSE_DISABLED);
    }
    if patches_sparse_seed {
        return Err(ROUTE_SPARSE_SEED);
    }
    if dense_unstable_rows_seed {
        return Err(ROUTE_DENSE_UNSTABLE_ROWS);
    }
    let CrownBounds::Patches(pb) = crown_bounds else {
        return Err(ROUTE_DENSE_CARRIER);
    };
    if pb.lower_a.unstable_idx.is_some() || pb.upper_a.unstable_idx.is_some() {
        return Err(ROUTE_SPARSE_LAYOUT);
    }
    if !matches!(
        (&pb.lower_a.geometry, &pb.upper_a.geometry),
        (PatchGeometry::Affine(_), PatchGeometry::Affine(_))
    ) {
        return Err(ROUTE_ANCHORED_GEOMETRY);
    }
    Ok(&**pb)
}

/// Largest per-row tap count a Patches side can visit: `in_c * kH * kW` for a
/// materialized tensor (the trailing three axes of every layout), 1 for the
/// virtual identity.
fn patch_taps_per_row(data: &crate::bounds::patches::PatchesData) -> usize {
    match data.patches.as_ref() {
        Some(patches) if patches.ndim() >= 3 => {
            let shape = patches.shape();
            shape[shape.len() - 3..].iter().product()
        }
        _ => 1,
    }
}

/// The route's own engagement observable, printed ONCE per process on the
/// first successful patch-native terminal publication. A replay that expects
/// the speedup must find this line or the run is VOID, not negative.
fn announce_patch_native_terminal_engaged(pb: &PatchesLinearBounds) {
    static ANNOUNCED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ANNOUNCED.get_or_init(|| {
        let (in_c, in_h, in_w) = pb.lower_a.input_shape;
        let in_dim = in_c.saturating_mul(in_h).saturating_mul(in_w);
        let taps_max = patch_taps_per_row(&pb.lower_a)
            .max(patch_taps_per_row(&pb.upper_a))
            .min(in_dim);
        let dense_pair_bytes = pb.dense_pair_bytes().unwrap_or(usize::MAX);
        eprintln!(
            "[crown-partial-pnc] ENGAGED rows={} in_dim={in_dim} taps_max={taps_max} \
             dense_pair_bytes={dense_pair_bytes} coeff_err={} (patch-native terminal \
             concretization replaces the densify on in-budget full-grid Patches terminals)",
            pb.row_count,
            u8::from(pb.lower_a.coeff_err.is_some() || pb.upper_a.coeff_err.is_some()),
        );
    });
}

/// Per-walk phase split and deciding route (unconditional stderr, slow walks
/// only). `walk` is the target layer index (`layers.len() - 1`).
fn report_partial_walk(
    walk: usize,
    backward_secs: f64,
    dense_secs: f64,
    concretize_secs: f64,
    route: &str,
) {
    if backward_secs + dense_secs + concretize_secs >= PNC_WALK_REPORT_SECS {
        eprintln!(
            "[crown-partial-pnc] walk={walk} backward_s={backward_secs:.3} \
             dense_s={dense_secs:.3} concretize_s={concretize_secs:.3} route={route}"
        );
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only trace of every terminal route decision made on this thread:
    /// `(route, terminal carried coeff_err)`. The partial walk runs on the
    /// caller's thread, so a test reads exactly its own decisions; a walk the
    /// GPU route answered records nothing, which a test must treat as VOID.
    static TERMINAL_ROUTE_TRACE: std::cell::RefCell<Vec<(&'static str, bool)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[inline]
fn record_terminal_route(route: &'static str, carried_err: bool) {
    #[cfg(test)]
    TERMINAL_ROUTE_TRACE.with(|trace| trace.borrow_mut().push((route, carried_err)));
    #[cfg(not(test))]
    let _ = (route, carried_err);
}

/// Drain this thread's terminal-route trace (test only).
#[cfg(test)]
fn take_terminal_route_trace() -> Vec<(&'static str, bool)> {
    TERMINAL_ROUTE_TRACE.with(|trace| std::mem::take(&mut *trace.borrow_mut()))
}

/// #crown-partial-finite-expiry — BUG #18 root cause D, the last two
/// deadline-PRESENCE declines of the whole-network-IBP set
/// (docs/DEADLINE_PRESENCE_FIX_2026-08-19.md).
///
/// Their set-mates in `sequential/crown/backward_step.rs`
/// (`finite_authority_refuses_per_layer_ibp`, the per-layer IBP salvage arms)
/// already decide by EXPIRY unconditionally, so with this lever armed the
/// COMPLETE root-cause-D set is on expiry — which is the only configuration
/// the audit predicts can measure as nonzero ("set-mates switch together;
/// partial fixes measure exactly zero").
///
/// Exact `"1"` arms; any other token (or absence) leaves the historical
/// presence guard byte-identical. Default OFF: unlike `NY_PATCHES_FINITE_EXPIRY`
/// these two sites have no measurement yet, and an unmeasured default flip is
/// how this campaign has manufactured fake results before.
///
/// The token contract is now owned by the typed declaration
/// `ny_levers::decls::diagnostics::CROWN_PARTIAL_FINITE_EXPIRY`, whose
/// `LeverKind::Bool` parser is the same exact-`"1"` rule this site always had:
/// `"0"` is an explicit disarm, every other spelling is a recorded rejection
/// that leaves the dark default, and a non-UTF-8 value cannot arm it — exactly
/// what the old `OsStr == "1"` comparison did, with the rejection now visible in
/// the flight receipt instead of being silently swallowed.
///
/// Latched once per process — this predicate sits on the per-node partial
/// walk, where a live environment read is a lock plus a scan of the environment
/// block (same idiom as `patches_step::expiry_authority_armed`).
fn crown_partial_expiry_armed() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| {
        // DELIVERY SEAM (#lever-reachability): `NY_CROWN_PARTIAL_FINITE_EXPIRY`
        // still wins in BOTH directions where present; absent, the typed
        // `bab.crown_partial_finite_expiry` key decides. The audit predicts a
        // null unless the WHOLE root-cause-D set switches together, so this
        // needed a route that can carry the set on a scored run rather than
        // only in a shell.
        let lever = crate::preset_gates::crown_partial_finite_expiry_enabled();
        // #cgan-reclaim item 3 (docs/CGAN_BUDGET_RECLAIM_2026-08-20.md): the
        // cgan reclaim set arms the SAME expiry semantics for this pair — the
        // measured live #18 signature on the scored cgan route is exactly this
        // pair's presence-decline (`crown-partial-salvage: saved=0
        // discarded=32` in every 900 s run). Route/seed selection only; both
        // arms produce sound enclosures, and the sound-GPU gate (default ON)
        // still refuses any unsound GPU backward from deciding a bound.
        let reclaim = crate::cgan_reclaim::cgan_reclaim_armed();
        let armed = lever || reclaim;
        if armed {
            // R9: the lever's own engagement announcement. A null measured
            // without this line is indistinguishable from an inert lever.
            eprintln!(
                "[deadline-preserve] crown-partial expiry ARMED (source: {}): CROWN-IBP \
                 partial GPU route and sparse seed discovery now decline on EXPIRY, \
                 not on deadline presence",
                if lever {
                    "NY_CROWN_PARTIAL_FINITE_EXPIRY"
                } else {
                    "NY_CGAN_RECLAIM"
                },
            );
        }
        armed
    })
}

/// Pure decision core, testable without touching the process environment or
/// the wall clock. `true` = decline (take the historical degraded arm).
///
/// * no deadline: never declines — byte-identical to the no-deadline tree;
/// * unarmed: declines on PRESENCE, exactly the historical guard;
/// * armed: declines only at or past expiry — commit-what-you-have semantics,
///   never extending past the deadline (the walk's existing entry/exit expiry
///   checks still refuse any late publication).
fn finite_authority_declines_partial(armed: bool, deadline: Option<Instant>, now: Instant) -> bool {
    deadline.is_some_and(|limit| !armed || now >= limit)
}

// ---------------------------------------------------------------------------
// #dense-sparse-seed — the Dense unstable-rows seed at alpha-beta-CROWN's
// threshold
// ---------------------------------------------------------------------------
//
// WHY. A Dense-carrier target (1D output: a Linear/Gemm, including a Gemm with
// its BatchNormalization folded in) seeds its backward walk with one row per
// IBP-unstable output neuron only when at least 90% of the neurons are
// IBP-stable (`find_unstable_dense_indices(.., 0.9)`).
// alpha-beta-CROWN takes that path whenever AT MOST 90% are unstable
// (`minimum_sparsity: 0.9`, `bound_general.py`), i.e. `min_stable = 0.1` here.
// On challenging_certified_training_2026 `cifar10_eps2_cnn7` the FC target
// (Gemm 32768->512 + BN) is ~86.7% stable (68/512 unstable), so the 0.9 rule
// walked all 512 rows back through five convolutions (6.2 s on the Mac DIAG
// log) and the root collection never finished inside the 30 s budget.
//
// WHAT. The typed key `bab.crown_partial_dense_min_stable` (declaration
// `NY_CROWN_PARTIAL_DENSE_MIN_STABLE`, default 0.9 = the historical literal)
// replaces that literal, but ONLY for a walk whose successor layer is a ReLU.
// Every other Dense walk (the network's final layer, a Sigmoid/Tanh successor)
// keeps 0.9, and the Patches conv branch and the GPU partial route keep their
// own 0.9 untouched.
//
// SOUNDNESS. No new code path: the seed, the backward walk over a subset of
// rows and the certified-IBP scatter (`scatter_sparse_crown_into_ibp`) are the
// production path every target that clears 0.9 already takes. The tracked set
// is the CERTIFIED-IBP-unstable set, a superset of the truly unstable neurons,
// and each untracked neuron publishes the certified IBP box verbatim, so every
// published value is an enclosure. Tracked rows walk the same per-row
// backward as under the full seed, so every neuron this box leaves unstable
// publishes the same value in both arms.
//
// NOT TIGHTNESS-NEUTRAL, IN EITHER DIRECTION. The two arms
// publish DIFFERENT operands for an untracked neuron. The full seed publishes
// raw CROWN, which the collector (`crown_ibp.rs`) intersects with its INITIAL
// forward IBP box (`ibp_bounds[k]`, propagated from the input). The sparse
// seed publishes this walk's `output_bounds` instead: the certified IBP of the
// target taken over the PUBLISHED, already-tightened prior. Neither operand
// contains the other. So an untracked neuron's magnitude can come out looser
// OR tighter, and a neuron the full seed would publish as unstable (CROWN and
// the initial IBP both straddle zero) can come out stable. The successor
// ReLU's relaxation is therefore equal or tighter. It is never looser (up to
// GEMM rounding on the tracked rows), because every tracked neuron keeps the
// full seed's CROWN value and every untracked one is sign-stable in
// `output_bounds`. Later targets see IBP operands chained from the changed
// magnitudes, so their bounds, their stability and hence their relaxations
// can move in either direction. Every value is still a certified enclosure:
// sound either way, verdict-affecting either way.
//
// OBSERVABLES (unconditional stderr, once per process each):
//   * `[dense-sparse-seed] ARMED min_stable=…` — the first time a walk reads a
//     resolved threshold that differs from the historical 0.9;
//   * `[dense-sparse-seed] ENGAGED rows=<tracked>/<total> walk=<target> …` —
//     the first walk whose sparse seed the historical 0.9 would NOT have
//     chosen, i.e. the first engagement attributable to this key. A run that
//     expects the speedup must find this line or it is VOID, not negative.
//   The per-walk `[crown-partial-pnc] walk=<k> … route=dense-unstable-rows`
//   line (#crown-partial-pnc) then carries that walk's phase timings.

/// Historical stability threshold of the Dense unstable-rows seed (#3599
/// Phase 2). Kept as the f32 literal the seed always used: the declaration
/// default `0.9_f64` narrows to exactly these bits (pinned by
/// `dense_min_stable_default_is_the_historical_literal` in ny-levers).
const DENSE_SEED_HISTORICAL_MIN_STABLE: f32 = 0.9;

/// Resolve `bab.crown_partial_dense_min_stable` for one walk: an admissible
/// environment value, then the typed preset cell, then the 0.9 default.
///
/// Deliberately NOT latched: the read happens once per Dense ReLU-successor
/// walk (a handful per collection, each costing milliseconds to seconds), and
/// an unlatched read cannot capture a test's transiently written cell for the
/// rest of a process.
fn dense_seed_min_stable_resolved() -> f32 {
    let resolved = crate::preset_gates::crown_partial_dense_min_stable();
    // The declaration admits only finite values in [0, 1]; anything else is
    // unreachable through the gate and keeps the historical literal.
    let threshold = if resolved.is_finite() && (0.0..=1.0).contains(&resolved) {
        resolved as f32
    } else {
        DENSE_SEED_HISTORICAL_MIN_STABLE
    };
    if threshold.to_bits() != DENSE_SEED_HISTORICAL_MIN_STABLE.to_bits() {
        announce_dense_seed_armed(threshold);
    }
    threshold
}

/// Does the walk over `layers` feed a ReLU? `layers` must be a PREFIX of
/// `all_layers` (the collector passes `&network.layers[0..=k]`); anything else
/// answers `false` and keeps the historical threshold.
fn walk_has_relu_successor(all_layers: &[Layer], layers: &[Layer]) -> bool {
    layers.len() < all_layers.len()
        && std::ptr::eq(all_layers.as_ptr(), layers.as_ptr())
        && matches!(all_layers.get(layers.len()), Some(Layer::ReLU(_)))
}

/// Pure: the stability threshold one Dense walk seeds with. The typed key is
/// consulted ONLY for a ReLU-successor walk; every other walk keeps 0.9
/// without resolving anything.
fn dense_seed_threshold(relu_successor: bool, armed: impl FnOnce() -> f32) -> f32 {
    if relu_successor {
        armed()
    } else {
        DENSE_SEED_HISTORICAL_MIN_STABLE
    }
}

/// Would the historical 0.9 threshold have chosen the sparse seed for
/// `unstable` IBP-unstable neurons out of `total`? Same f32 arithmetic as
/// `find_unstable_dense_indices`, so "attributable to the key" is exact.
fn historical_threshold_seeds_sparse(unstable: usize, total: usize) -> bool {
    if total == 0 || unstable == 0 {
        return false;
    }
    let stable_frac = 1.0 - (unstable as f32 / total as f32);
    stable_frac >= DENSE_SEED_HISTORICAL_MIN_STABLE
}

fn announce_dense_seed_armed(threshold: f32) {
    static ANNOUNCED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ANNOUNCED.get_or_init(|| {
        eprintln!(
            "[dense-sparse-seed] ARMED min_stable={threshold:.3} \
             (bab.crown_partial_dense_min_stable; historical \
             {DENSE_SEED_HISTORICAL_MIN_STABLE}): Dense-seeded CROWN-IBP partial walks \
             with a ReLU successor seed only their IBP-unstable rows once this fraction \
             of their neurons is IBP-stable"
        );
    });
}

/// The key's own engagement observable, printed ONCE per process on the first
/// sparse seed the historical 0.9 threshold would not have chosen.
fn announce_dense_seed_engaged(walk: usize, rows: usize, total: usize, threshold: f32) {
    static ANNOUNCED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ANNOUNCED.get_or_init(|| {
        let stable_frac = 1.0 - rows as f64 / total.max(1) as f64;
        eprintln!(
            "[dense-sparse-seed] ENGAGED rows={rows}/{total} walk={walk} \
             stable_frac={stable_frac:.3} min_stable={threshold:.3} (historical \
             {DENSE_SEED_HISTORICAL_MIN_STABLE} would have seeded all {total} rows; \
             untracked rows publish the certified IBP box)"
        );
    });
}

/// One Dense seed decision (test only): which target, how many rows it seeded
/// out of how many, at which threshold, and whether the historical 0.9 would
/// have made a different choice.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct DenseSeedDecision {
    walk: usize,
    rows: usize,
    total: usize,
    min_stable: f32,
    attributable: bool,
}

#[cfg(test)]
thread_local! {
    /// Test-only trace of every Dense seed decision on this thread (the walk
    /// runs on the caller's thread). A walk the GPU route answered records
    /// nothing, which a test must treat as VOID.
    static DENSE_SEED_TRACE: std::cell::RefCell<Vec<DenseSeedDecision>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[inline]
fn record_dense_seed(walk: usize, rows: usize, total: usize, min_stable: f32, attributable: bool) {
    #[cfg(test)]
    DENSE_SEED_TRACE.with(|trace| {
        trace.borrow_mut().push(DenseSeedDecision {
            walk,
            rows,
            total,
            min_stable,
            attributable,
        });
    });
    #[cfg(not(test))]
    let _ = (walk, rows, total, min_stable, attributable);
}

/// Drain this thread's Dense seed trace (test only).
#[cfg(test)]
fn take_dense_seed_trace() -> Vec<DenseSeedDecision> {
    DENSE_SEED_TRACE.with(|trace| std::mem::take(&mut *trace.borrow_mut()))
}

/// [deadline-preserve] saved-vs-discarded counts for the root-cause-D pair.
/// Two decisions are recorded per deadlined partial walk (GPU route, sparse
/// seed); recording happens after each decision and changes nothing.
static CROWN_PARTIAL_PRESERVE:
    crate::network::core::graph::backward_helpers::DeadlinePreserveCounters =
    crate::network::core::graph::backward_helpers::DeadlinePreserveCounters::new(
        "crown-partial-salvage",
    );

/// Walk-time attribution by layer kind for the sequential partial backward
/// (#budget-log companion; read by the `[walk-share]` diagnostics line).
pub(crate) static WALK_CONV_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static WALK_ACT_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static WALK_OTHER_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static WALK_FINAL_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Propagate CROWN bounds through a partial network (subset of layers).
///
/// Uses Patches-aware backward dispatch via [`crown_backward_step_patches`]
/// to avoid materializing dense A-matrices for Conv2d layers. When the output
/// is 3D spatial and the sub-network contains Conv2d, bounds are initialized
/// in Patches mode (O(out_c * kH * kW) per spatial position) instead of
/// Dense mode (O(out_dim²) total).
///
/// Production entry: the partial walk with the terminal route fixed by
/// [`PATCH_NATIVE_TERMINAL_CONCRETIZE`] (#crown-partial-pnc).
pub(super) fn propagate_crown_partial_with_engine(
    all_layers: &[Layer],
    input: &BoundedTensor,
    layers: &[Layer],
    prior_bounds: &[BoundedTensor],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    patches_deadline_flat_bias_enabled: Option<bool>,
) -> Result<PartialCrownPropagationResult> {
    propagate_crown_partial_routed(
        all_layers,
        input,
        layers,
        prior_bounds,
        engine,
        deadline,
        patches_deadline_flat_bias_enabled,
        PATCH_NATIVE_TERMINAL_CONCRETIZE,
        None,
    )
}

/// The partial walk with an explicit terminal route. `patch_native_terminal`
/// only selects HOW an in-scope terminal Patches carrier is concretized (see
/// the #crown-partial-pnc block above); tests drive both arms through this
/// one body to pin bit-identity.
///
/// `dense_min_stable` (#dense-sparse-seed) injects the Dense unstable-rows
/// seed threshold for a ReLU-successor walk; `None` (production) resolves
/// `bab.crown_partial_dense_min_stable` lazily, only when such a walk needs it.
/// A walk without a ReLU successor ignores it and keeps the historical 0.9.
#[allow(clippy::too_many_arguments)]
fn propagate_crown_partial_routed(
    all_layers: &[Layer],
    input: &BoundedTensor,
    layers: &[Layer],
    prior_bounds: &[BoundedTensor],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    patches_deadline_flat_bias_enabled: Option<bool>,
    patch_native_terminal: bool,
    dense_min_stable: Option<f32>,
) -> Result<PartialCrownPropagationResult> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP partial: deadline exceeded before entry".to_string(),
        ));
    }
    if layers.is_empty() {
        return Ok(PartialCrownPropagationResult::Crown(Box::new(
            input.clone(),
        )));
    }

    // The last layer's per-layer box. #crown-partial-sound-forward: this box is
    // not "shape only" — it selects the sparse/dense seed rows, is scattered
    // VERBATIM into every untracked row of the returned CROWN-IBP bound
    // (`merge_sparse_crown_with_ibp`, `scatter_sparse_crown_into_ibp`) and is
    // the IBP operand of the GPU route's intersection, and the collector then
    // publishes the result into a verdict. It used to be the PLAIN
    // `Layer::propagate_ibp` (round-to-nearest f32 on the conv family, no
    // Higham term, documented INWARD under cancellation — see
    // `conv2d/bound.rs` `propagate_ibp_sound_with_engine`). It is now the
    // certified forward, and every site that consumes it takes only the
    // `CertifiedIbpBox` type. The pre-last chain below (taken only when the
    // caller supplied fewer prior bounds than layers — dead from the
    // collector, which always passes exactly `k` entries) is certified too.
    let last_layer = layers.last().ok_or_else(|| {
        NyError::InvalidSpec("CROWN partial propagation requires at least one layer".to_string())
    })?;
    let pre_last = if layers.len() == 1 {
        input.clone()
    } else if prior_bounds.len() >= layers.len() - 1 {
        prior_bounds[layers.len() - 2].clone()
    } else {
        // Compute the intermediate box if not available — certified per layer.
        let mut current = input.clone();
        for layer in layers.iter().take(layers.len() - 1) {
            current = propagate_ibp_certified(layer, &current)?.into_tensor();
        }
        current
    };
    let output_bounds = propagate_ibp_certified(last_layer, &pre_last)?;
    let output_dim = output_bounds.as_tensor().len();
    let output_shape = output_bounds.as_tensor().shape().to_vec();

    // GPU fast path: dispatch the entire per-node backward to GPU
    // when all layers in the sub-network are GPU-extractable. This avoids the
    // CPU-bound Patches/Dense backward loop and host-side concretization.
    // Historical stance, now lever-decided below: a finite request
    // deliberately skipped this optional GPU route. Although a backend can
    // advertise cooperative device cancellation, the legacy host preparation
    // still extracts/copies every layer and input endpoint and builds the
    // full (or sparse) specification matrix without cooperative polls. The
    // CPU Patches/Dense path below observes the same absolute deadline
    // through its admitted materialization boundaries.
    // Soundness gate (#vnncomp-gpu-crown-soundness): under the gate, route to the
    // SOUND GPU-resident backward (`use_sound = true`) when the engine advertises
    // it — that path carries the certified `γ_n·S` coefficient-rounding error
    // through every on-device AW GEMM, so this verdict-relevant INTERMEDIATE CROWN
    // bound is a sound enclosure decided on GPU instead of bailing to the CPU sound
    // loop. Without the gate (`use_sound = false`) the existing fast unsound path
    // runs. Either way an `Err`/NaN inside falls back to the proven CPU loop below.
    // See `sound_gpu_gate`.
    // #crown-partial-finite-expiry set-mate 1/2 (BUG #18 root cause D). The
    // historical guard above ("deliberately skips") tested deadline PRESENCE,
    // so this route ran only in tests — every scored run carries a deadline.
    // The route is already deadline-plumbed for a live limit: the entry/exit
    // expiry checks below return `DeadlineExceeded` instead of publishing a
    // late result, dispatch is bracketed by `GpuCrownBackendDeadlineScope`,
    // and `gpu_crown_backend_honors_deadline` refuses any backend that cannot
    // observe the limit. Armed, a live unexpired deadline keeps the route;
    // expired or unarmed restores the historical decline byte-identically.
    // Wall-clock safety unchanged: nothing here extends the deadline. Sound-
    // ness: route selection only — both arms produce sound enclosures and no
    // bound value is created at this site. Bounded residue, stated: the host
    // layer extraction inside `try_gpu_crown_partial_backward` is not
    // internally polled, so an armed run can overrun by that one preparation
    // phase — the same accepted tradeoff class as NY_PATCHES_FINITE_EXPIRY.
    let gpu_partial_declined =
        finite_authority_declines_partial(crown_partial_expiry_armed(), deadline, Instant::now());
    if deadline.is_some() {
        CROWN_PARTIAL_PRESERVE.record(gpu_partial_declined);
    }
    let gpu_route = (!gpu_partial_declined)
        .then(|| crate::sound_gpu_gate::gpu_crown_backward_route_with_deadline(engine, deadline));
    if let Some(Some((gpu, use_sound))) = gpu_route {
        if crate::sound_gpu_gate::gpu_crown_backend_honors_deadline(gpu, deadline) {
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return Err(NyError::DeadlineExceeded(
                    "CROWN-IBP partial: deadline exceeded before GPU dispatch".to_string(),
                ));
            }
            let _gpu_deadline_scope =
                crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(gpu, deadline);
            let gpu_result = try_gpu_crown_partial_backward(
                layers,
                prior_bounds,
                input,
                gpu,
                use_sound,
                output_dim,
                &output_shape,
                &output_bounds,
                deadline,
            )?;
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return Err(NyError::DeadlineExceeded(
                    "CROWN-IBP partial: deadline exceeded after GPU dispatch".to_string(),
                ));
            }
            if let Some(gpu_result) = gpu_result {
                return Ok(gpu_result);
            }
        }
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP partial: deadline exceeded before CPU fallback".to_string(),
        ));
    }

    // Initialize CROWN bounds — use Patches mode when the output is 3D spatial
    // (channels, height, width) and the sub-network contains Conv2d. This avoids
    // materializing dense A-matrices of size O(out_dim²) in favor of sparse
    // patch coefficients of size O(out_c * kH * kW) per spatial position.
    //
    // Sparse patches optimization: when IBP bounds show >90%
    // of output neurons are stable (all positive or all negative), only track
    // unstable neurons in the backward pass. This reduces memory from
    // O(total * receptive_field) to O(unstable * receptive_field).
    let has_conv2d = layers.iter().any(|l| matches!(l, Layer::Conv2d(_)));
    let is_sparse_mode;
    // Phase 2: track unstable indices for Dense sparse mode.
    // When set, only unstable neurons have spec rows in the backward pass.
    let mut dense_unstable_indices: Option<Vec<usize>> = None;
    // Flat output positions the SPARSE PATCHES seed actually tracks. The merge
    // below must know these positionally: an untracked neuron's concretized
    // bound is `[bias, bias]`, which is indistinguishable by value from a
    // tracked neuron that legitimately lands there. See
    // `merge_sparse_crown_with_ibp`.
    let mut patches_tracked_flat: Option<Vec<usize>> = None;
    let mut crown_bounds = if has_conv2d && output_shape.len() == 3 {
        let spatial = (output_shape[0], output_shape[1], output_shape[2]);
        // Try sparse patches from IBP output bounds.
        // Historical stance: sparse discovery and seed construction retain
        // legacy infallible `Vec` growth and full endpoint scans, so finite
        // authority was kept on the admitted full virtual-identity route.
        // That stance is now lever-decided below.
        // #crown-partial-finite-expiry set-mate 2/2 (BUG #18 root cause D).
        // Sparse discovery is one O(output) endpoint scan plus index-Vec
        // growth — bounded, front-loaded work. Declining it on deadline
        // PRESENCE forced every scored run onto the FULL identity seed, the
        // strictly larger allocation, which is exactly the degradation this
        // set removes. The sparse seed tracks a subset of the full seed's
        // rows, so its footprint is bounded by the full seed the historical
        // arm admits; untracked rows merge from the CERTIFIED IBP box
        // (`merge_sparse_crown_with_ibp`, typed `CertifiedIbpBox` since
        // #crown-partial-sound-forward — before that it was the plain box, and
        // "the SAME sound IBP bounds" here was false), so seed choice never
        // manufactures a value — numerics match the no-deadline arm bit for
        // bit. Expired or unarmed: exact historical behavior.
        let sparse_declined = finite_authority_declines_partial(
            crown_partial_expiry_armed(),
            deadline,
            Instant::now(),
        );
        if deadline.is_some() {
            CROWN_PARTIAL_PRESERVE.record(sparse_declined);
        }
        let sparse_patches = if !sparse_declined {
            crate::bounds::patches::UnstableIdx::from_ibp_bounds(
                output_bounds.as_tensor().lower().as_slice().unwrap_or(&[]),
                output_bounds.as_tensor().upper().as_slice().unwrap_or(&[]),
                spatial,
                0.9,
            )
        } else {
            None
        };
        if let Some(unstable_idx) = sparse_patches {
            let n = unstable_idx.len();
            let total = spatial.0 * spatial.1 * spatial.2;
            debug!(
                "CROWN-IBP partial: Sparse Patches mode — {}/{} unstable ({:.1}% sparse), {:?}",
                n,
                total,
                (1.0 - n as f64 / total as f64) * 100.0,
                spatial,
            );
            is_sparse_mode = true;
            // Capture the tracked positions BEFORE the seed consumes the index
            // set; the merge after concretization needs them.
            patches_tracked_flat = Some(
                (0..unstable_idx.len())
                    .map(|i| unstable_idx.flat_index(i, spatial.1, spatial.2))
                    .collect(),
            );
            CrownBounds::Patches(Box::new(PatchesLinearBounds::sparse_identity(
                spatial,
                spatial,
                unstable_idx,
            )))
        } else {
            debug!(
                "CROWN-IBP partial: Patches mode — 3D spatial output {:?} with Conv2d",
                spatial
            );
            is_sparse_mode = false;
            let seed = match PatchesLinearBounds::try_identity_with_deadline(
                spatial, spatial, deadline, 0,
            ) {
                Ok(seed) => seed,
                Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                    return Ok(PartialCrownPropagationResult::ForwardFallback(
                        CrownStepFallback {
                            reason: crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded,
                            details: format!(
                                "CROWN-IBP partial identity seed exceeded its CPU budget: {error}"
                            ),
                        },
                    ));
                }
                Err(error) => return Err(error),
            };
            CrownBounds::Patches(Box::new(seed))
        }
    } else {
        is_sparse_mode = false;

        // Phase 2: try unstable-only Dense specs. If at least
        // `min_stable` of the output neurons are stable (historically >= 90%),
        // build a reduced (n_unstable × output_dim) identity instead of the
        // full (output_dim × output_dim). Reduces GEMM work proportionally to
        // the stability ratio.
        //
        // Reference: alpha-beta-CROWN backward_bound.py uses sparse C matrices
        // sized by unstable neuron count for the same optimization.
        //
        // #dense-sparse-seed: `min_stable` is the typed key's value on a
        // ReLU-successor walk and the historical 0.9 on every other walk (see
        // the block above `dense_seed_threshold`).
        let min_stable = dense_seed_threshold(walk_has_relu_successor(all_layers, layers), || {
            dense_min_stable.unwrap_or_else(dense_seed_min_stable_resolved)
        });
        let unstable_idx = find_unstable_dense_indices(
            output_bounds.as_tensor().lower().as_slice().unwrap_or(&[]),
            output_bounds.as_tensor().upper().as_slice().unwrap_or(&[]),
            min_stable,
        );
        {
            let walk = layers.len() - 1;
            let (rows, attributable) = match unstable_idx.as_ref() {
                Some(idx) => (
                    idx.len(),
                    !historical_threshold_seeds_sparse(idx.len(), output_dim),
                ),
                None => (output_dim, false),
            };
            if attributable {
                announce_dense_seed_engaged(walk, rows, output_dim, min_stable);
            }
            record_dense_seed(walk, rows, output_dim, min_stable, attributable);
        }

        let seed_rows = if let Some(ref idx) = unstable_idx {
            debug!(
                "CROWN-IBP partial: Dense sparse mode — {}/{} unstable ({:.1}% sparse)",
                idx.len(),
                output_dim,
                (1.0 - idx.len() as f64 / output_dim as f64) * 100.0,
            );
            idx.len()
        } else {
            output_dim
        };

        // Memory estimate uses actual seed rows (potentially reduced).
        let required_bytes = seed_rows
            .checked_mul(output_dim)
            .and_then(|n: usize| n.checked_mul(2 * size_of::<f32>()))
            .unwrap_or(usize::MAX);
        let estimate = DenseMaterializationEstimate {
            site: "initial_dense_identity",
            rows: seed_rows,
            cols: output_dim,
            required_bytes,
        };
        if estimate.exceeds_budget(cpu_crown_dense_budget_bytes()) {
            return Ok(PartialCrownPropagationResult::ForwardFallback(
                memory_budget_partial_fallback(estimate, "CROWN-IBP partial"),
            ));
        }

        let seed = if let Some(ref idx) = unstable_idx {
            // Sparse identity: each row selects one unstable output neuron.
            let n = idx.len();
            let mut lower_a = Array2::zeros((n, output_dim));
            let mut upper_a = Array2::zeros((n, output_dim));
            for (row, &col) in idx.iter().enumerate() {
                lower_a[(row, col)] = 1.0;
                upper_a[(row, col)] = 1.0;
            }
            LinearBounds {
                lower_a,
                lower_b: Array1::zeros(n),
                upper_a,
                upper_b: Array1::zeros(n),
                // Exact sparse-identity selection: no accumulated f32 error.
                lower_a_err: None,
                upper_a_err: None,
            }
        } else {
            LinearBounds::identity(output_dim)
        };
        dense_unstable_indices = unstable_idx;
        CrownBounds::Dense(seed)
    };

    // Propagate backward through each layer using Patches-aware dispatch.
    let backward_start = Instant::now();
    for (i, layer) in layers.iter().enumerate().rev() {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "CROWN-IBP partial: per-node deadline exceeded at layer {} ({})",
                i,
                layer.layer_type()
            )));
        }

        // Get pre-activation bounds for this layer.
        let pre_activation = if i == 0 {
            input
        } else if i - 1 < prior_bounds.len() {
            &prior_bounds[i - 1]
        } else {
            return Err(NyError::InvalidSpec(
                "Missing prior bounds for CROWN-IBP".to_string(),
            ));
        };

        let step_t = Instant::now();
        let step_result = match patches_deadline_flat_bias_enabled {
            Some(flat_bias_enabled) => {
                crown_backward_step_patches_with_deadline_authority_and_flat_bias_policy(
                    layer,
                    &mut crown_bounds,
                    pre_activation,
                    engine,
                    i,
                    "CROWN-IBP",
                    deadline,
                    deadline.is_some(),
                    flat_bias_enabled,
                )
            }
            None => crown_backward_step_patches(
                layer,
                &mut crown_bounds,
                pre_activation,
                engine,
                i,
                "CROWN-IBP",
                deadline,
            ),
        };
        // #budget-log companion: attribute walk time by layer kind so a
        // per-node overrun points at an arm, not at "the walk".
        let bucket = {
            let t = layer.layer_type();
            if t.contains("Conv") {
                &WALK_CONV_NANOS
            } else if t.contains("Relu") || t.contains("ReLU") || t.contains("Sigmoid") {
                &WALK_ACT_NANOS
            } else {
                &WALK_OTHER_NANOS
            }
        };
        bucket.fetch_add(
            u64::try_from(step_t.elapsed().as_nanos()).unwrap_or(u64::MAX),
            std::sync::atomic::Ordering::Relaxed,
        );
        match step_result? {
            CrownStepResult::Continue => {}
            CrownStepResult::IbpFallback(fallback) => {
                return Ok(PartialCrownPropagationResult::ForwardFallback(fallback));
            }
        }
    }
    let backward_secs = backward_start.elapsed().as_secs_f64();

    // Convert to Dense for concretization (Patches → Dense is a no-op if already Dense).
    let dense_start = Instant::now();
    if let Some(fallback) = patches_dense_materialization_fallback(
        &crown_bounds,
        "final_concretization",
        "CROWN-IBP partial",
    )? {
        return Ok(PartialCrownPropagationResult::ForwardFallback(fallback));
    }

    // #crown-partial-pnc: patch-native terminal concretization (see the block
    // at the top of this file). The over-budget refusal just above is
    // unchanged, so an in-scope carrier's dense pair is within budget. Only a
    // SUCCESSFUL patch-native publication returns from here; `DeadlineExceeded`
    // propagates as the dense route's would; every other refusal falls through
    // to the historical densify-and-concretize path below, byte-identical to
    // the pre-change tree.
    let walk = layers.len() - 1;
    let terminal_route = match patch_native_terminal_scope(
        patch_native_terminal,
        is_sparse_mode,
        dense_unstable_indices.is_some(),
        &crown_bounds,
    ) {
        Err(out_of_scope) => {
            record_terminal_route(out_of_scope, false);
            out_of_scope
        }
        Ok(pb) => {
            let carried_err = pb.lower_a.coeff_err.is_some() || pb.upper_a.coeff_err.is_some();
            let pnc_start = Instant::now();
            match pb.concretize_sound_sparse(input, deadline) {
                Ok(flat) => {
                    let crown_result = flat.into_reshape_with_poll(&output_shape, || {
                        check_partial_deadline(deadline)
                    })?;
                    let dense_secs = pnc_start.duration_since(dense_start).as_secs_f64();
                    let concretize_secs = pnc_start.elapsed().as_secs_f64();
                    let crown_result =
                        publish_concretized_crown(crown_result, deadline, Instant::now())?;
                    record_terminal_route(ROUTE_PATCH_NATIVE, carried_err);
                    announce_patch_native_terminal_engaged(pb);
                    report_partial_walk(
                        walk,
                        backward_secs,
                        dense_secs,
                        concretize_secs,
                        ROUTE_PATCH_NATIVE,
                    );
                    return Ok(PartialCrownPropagationResult::Crown(Box::new(crown_result)));
                }
                Err(error @ NyError::DeadlineExceeded(_)) => return Err(error),
                Err(error) => {
                    let route = if matches!(error, NyError::UnsupportedOp(_)) {
                        ROUTE_DENSE_UNSUPPORTED
                    } else {
                        ROUTE_DENSE_REFUSED
                    };
                    debug!(
                        "CROWN-IBP partial: patch-native terminal concretization declined \
                         ({route}), historical dense route decides: {error}"
                    );
                    record_terminal_route(route, carried_err);
                    route
                }
            }
        }
    };

    let Some(linear_bounds) =
        materialize_terminal_crown_bounds_with_deadline(crown_bounds, deadline)?
    else {
        return Ok(PartialCrownPropagationResult::ForwardFallback(
            CrownStepFallback {
                reason: crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded,
                details:
                    "CROWN-IBP partial final Patches materialization exceeded its full peak budget"
                        .to_string(),
            },
        ));
    };
    let dense_secs = dense_start.elapsed().as_secs_f64();
    WALK_FINAL_NANOS.fetch_add(
        u64::try_from(dense_start.elapsed().as_nanos()).unwrap_or(u64::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );

    // Concretize linear bounds with input bounds and reshape to the IBP output shape.
    //
    // LinearBounds::concretize returns a flat 1D tensor. For ONNX models (and any model
    // with non-1D activations), this causes a shape mismatch against IBP bounds and
    // prevents CROWN-IBP from intersecting/tightening intermediate bounds.
    let concretize_start = Instant::now();

    // Phase 2: Dense sparse mode — concretize gives n_unstable values,
    // not output_dim. Scatter CROWN bounds to unstable positions; use the
    // CERTIFIED IBP box for stable neurons where CROWN tightening has no
    // benefit (exact ReLU relaxation). This scatter is NOT gated by the
    // deadline lever, so it was the plain-box publication live on every
    // deadlined run before #crown-partial-sound-forward.
    if let Some(ref idx) = dense_unstable_indices {
        let sparse_crown = match linear_bounds.concretize_sound_with_deadline(input, deadline) {
            Ok(bounds) => bounds,
            Err(error) => {
                if let Some(fallback) = concretization_memory_fallback(&error) {
                    return Ok(PartialCrownPropagationResult::ForwardFallback(fallback));
                }
                return Err(error);
            }
        };
        let crown_lower = contiguous_flat_slice(sparse_crown.lower());
        let crown_upper = contiguous_flat_slice(sparse_crown.upper());
        let concretize_secs = concretize_start.elapsed().as_secs_f64();
        let partial_total = backward_secs + dense_secs + concretize_secs;
        if partial_total > 0.5 {
            debug!(
                "CROWN-IBP partial ({} layers, dense_sparse): backward={backward_secs:.3}s dense_conv={dense_secs:.3}s concretize={concretize_secs:.3}s total={partial_total:.3}s",
                layers.len(),
            );
        }
        report_partial_walk(
            walk,
            backward_secs,
            dense_secs,
            concretize_secs,
            terminal_route,
        );
        let crown_result = scatter_sparse_crown_into_ibp(
            &crown_lower,
            &crown_upper,
            &output_bounds,
            idx,
            &output_shape,
        )?;
        let crown_result = publish_concretized_crown(crown_result, deadline, Instant::now())?;
        return Ok(PartialCrownPropagationResult::Crown(Box::new(crown_result)));
    }

    let crown_result = match linear_bounds.concretize_sound_with_deadline(input, deadline) {
        Ok(bounds) => bounds,
        Err(error) => {
            if let Some(fallback) = concretization_memory_fallback(&error) {
                return Ok(PartialCrownPropagationResult::ForwardFallback(fallback));
            }
            return Err(error);
        }
    };
    let crown_result =
        crown_result.into_reshape_with_poll(&output_shape, || check_partial_deadline(deadline))?;
    let concretize_secs = concretize_start.elapsed().as_secs_f64();

    // Per-partial-pass timing: log when significant.
    let partial_total = backward_secs + dense_secs + concretize_secs;
    if partial_total > 0.5 {
        let mode = if is_sparse_mode {
            "sparse"
        } else if has_conv2d && output_shape.len() == 3 {
            "patches"
        } else {
            "dense"
        };
        debug!(
            "CROWN-IBP partial ({} layers, {mode}): backward={backward_secs:.3}s dense_conv={dense_secs:.3}s concretize={concretize_secs:.3}s total={partial_total:.3}s",
            layers.len(),
        );
    }
    report_partial_walk(
        walk,
        backward_secs,
        dense_secs,
        concretize_secs,
        terminal_route,
    );

    // Sparse mode merge: CROWN only computed bounds for unstable
    // neurons (sparse rows). Stable neurons got zero rows, producing [0, 0] bounds
    // after concretization. Replace those with the CERTIFIED IBP box, which is
    // already tight for stable neurons (all positive or all negative) and — since
    // #crown-partial-sound-forward — encloses the true range there.
    let crown_result = if is_sparse_mode {
        // Fail closed if the tracked-index set is somehow absent: without it the
        // merge cannot tell tracked from untracked, and guessing by value is the
        // defect this threading exists to remove.
        let tracked = patches_tracked_flat.as_deref().ok_or_else(|| {
            NyError::InternalError(
                "sparse CROWN merge: sparse mode without a tracked-index set".into(),
            )
        })?;
        merge_sparse_crown_with_ibp(&crown_result, &output_bounds, tracked)?
    } else {
        crown_result
    };
    let crown_result = publish_concretized_crown(crown_result, deadline, Instant::now())?;
    Ok(PartialCrownPropagationResult::Crown(Box::new(crown_result)))
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use ndarray::arr1;
    use std::time::Duration;

    #[test]
    fn valid_result_completed_after_deadline_is_rejected() {
        let valid = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
        let deadline = Instant::now();
        let completed_at = deadline + Duration::from_nanos(1);

        let error = match publish_concretized_crown(valid, Some(deadline), completed_at) {
            Ok(_) => panic!("a valid but late CROWN result must not be published"),
            Err(error) => error,
        };

        assert!(
            matches!(error, NyError::DeadlineExceeded(ref message)
                if message.contains("after concretization")),
            "late CROWN result should fail open as DeadlineExceeded, got {error:?}"
        );
    }
}

#[cfg(test)]
mod deadline_presence_fix_tests {
    use super::*;
    use std::time::Duration;

    // BUG #18 root cause D (docs/DEADLINE_PRESENCE_FIX_2026-08-19.md): the
    // decision core is pure, so the entire lever truth table is pinned here
    // GPU-free, independent of the process environment and the wall clock.

    #[test]
    fn no_deadline_never_declines() {
        let now = Instant::now();
        assert!(!finite_authority_declines_partial(false, None, now));
        assert!(!finite_authority_declines_partial(true, None, now));
    }

    #[test]
    fn unarmed_declines_on_presence_exactly_like_the_historical_guard() {
        let now = Instant::now();
        let live = now + Duration::from_mins(1);
        assert!(finite_authority_declines_partial(false, Some(live), now));
    }

    #[test]
    fn armed_keeps_a_live_deadline() {
        let now = Instant::now();
        let live = now + Duration::from_mins(1);
        assert!(!finite_authority_declines_partial(true, Some(live), now));
    }

    #[test]
    fn armed_still_declines_at_and_past_expiry() {
        // Wall-clock safety unchanged: arming never admits work at or past
        // the limit — commit-what-you-have applies only to budget still live.
        let limit = Instant::now();
        assert!(finite_authority_declines_partial(true, Some(limit), limit));
        assert!(finite_authority_declines_partial(
            true,
            Some(limit),
            limit + Duration::from_nanos(1)
        ));
    }

    /// The same truth table the now-deleted local `expiry_token_arms` predicate
    /// pinned, exercised against the TYPED DECLARATION that
    /// `crown_partial_expiry_armed` reads. `read_with` supplies the token
    /// directly, so this stays GPU-free and independent of the process
    /// environment, exactly as before.
    #[test]
    fn lever_token_is_exact_one() {
        let arms = |raw: Option<&str>| {
            ny_levers::read_with(
                &ny_levers::decls::diagnostics::CROWN_PARTIAL_FINITE_EXPIRY,
                |_| raw.map(str::to_owned),
            )
            .value
            .as_bool()
        };
        assert!(arms(Some("1")));
        assert!(!arms(Some("0")));
        assert!(!arms(Some("true")));
        assert!(!arms(Some("")));
        assert!(!arms(Some(" 1")));
        assert!(!arms(None));
    }

    #[test]
    fn preserve_counters_report_saved_vs_discarded() {
        let counters = crate::network::core::graph::backward_helpers::DeadlinePreserveCounters::new(
            "test-set",
        );
        counters.record(false);
        counters.record(false);
        counters.record(true);
        assert_eq!(counters.snapshot(), (2, 1));
    }
}

#[cfg(test)]
#[path = "crown_partial_pnc_tests.rs"]
mod pnc_tests;

#[cfg(test)]
#[path = "crown_partial_dense_seed_tests.rs"]
mod dense_seed_tests;
