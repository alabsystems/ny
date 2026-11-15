// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #backward-interm: backward-computed root intermediates for the margin-row
//! lane (#twinwall). The declaration default is OFF and the legacy override is
//! `NY_MARGIN_ROW_BACKWARD_INTERM`; measured CIFAR100 and TinyImageNet typed
//! presets explicitly arm it on scored runs.
//!
//! WHY. The lane's trunk gates come from the FORWARD (M, D) tableau
//! (`root.rs::concretize_box`) — interval-style propagation, the weakest of
//! the family, and the measured binding constraint on the cifar100 deep band
//! (lane roots −0.61..−1.76 where abc proves at INIT with zero BaB). This
//! module recomputes each trunk ReLU's INPUT box with the lane's OWN certified
//! backward engine — identity rows seeded at that layer's pre-activation, run
//! through the already-frozen PREFIX gates, concretized over the root box,
//! exactly like alpha-CROWN intermediate bounds — and INTERSECTS (shrink-only)
//! with the forward box BEFORE `gates_from_box` derives `(alpha, s, c, ms)`.
//! It must run DURING the forward build, layer by layer: layer `li`'s backward
//! pass consumes the gates of layers `0..li`, which the tightening of earlier
//! layers has already improved, so the effect COMPOUNDS down the trunk.
//!
//! SOUNDNESS.
//! * Both boxes are valid outward enclosures of the true pre-activation range
//!   over the root box: the forward one by the tableau's certified D-lane, the
//!   backward one by the engine's certified error carry + outward concretize
//!   (`engine.rs::run_prefix` docs). The intersection of two valid enclosures
//!   is a valid enclosure; `max` of lowers / `min` of uppers is shrink-only.
//! * Two sound enclosures of a NON-EMPTY true range cannot cross. A crossed
//!   intersection therefore indicates a defect somewhere — it is NOT treated
//!   as an infeasibility certificate (this is the ROOT box; "empty" would be
//!   absurd and acting on it a false-UNSAT). The neuron keeps its forward
//!   bounds and the event is counted + printed loudly.
//! * Downstream, `gates_from_box` + `repair_upper_lines` re-certify the gates
//!   on the PUBLISHED (tightened) box, which contains the true range; the
//!   chord obligation `s*y + c >= relu(y)` is checked on that box's kinks as
//!   always. Skipping any layer, chunk, or neuron is always sound (the forward
//!   box simply stands).
//! * THE ONE ORDERING TRAP: `LayerGates::clip_rows` slack calibration must
//!   recover the LINE's own certified slack (`lo_min - l_published`). With a
//!   tightened `l` published, that difference under-recovers the slack (can go
//!   negative → clamped to 0), and every Clip-and-Verify halfspace built from
//!   the line would cut into the true subdomain — a false-`unsat` generator.
//!   `root.rs` therefore calibrates against the FORWARD-only bounds, captured
//!   before this module runs (see the `bi_cal` comment at the retention site).
//!
//! COST/GATING. One Lower + one Upper prefix pass per column chunk per layer.
//! Budgeted: `NY_MARGIN_ROW_BI_SECS` (default 20 s) further capped to 40% of
//! the remaining deadline (the same starvation guard as `alpha_opt`), checked
//! before every chunk. `NY_MARGIN_ROW_BI_TOPK` (default 1024) selects the
//! widest unstable neurons per layer; `NY_MARGIN_ROW_BI_CHUNK` (default 256)
//! bounds the pass width (memory is `O(max_tensor * chunk)`). Layer 0 is
//! skipped: its prefix is purely affine, so the forward tableau there is
//! already the exact linear enclosure and a backward pass can only reproduce
//! it. v1 runs only on unsplit root builds (`splits.is_empty()`), never inside
//! Tier-2 epoch rebuilds. The base root pays once; explicitly governed rebake
//! rounds may run separately re-budgeted root builds.
//!
//! THE TWO CLOCKS — read this before quoting any `secs=` from this phase.
//! `BiState::t0` starts immediately before the forward tableau op walk, after
//! root setup and validation, not when this phase's first pass starts
//! (`tighten_layer` is called from inside that walk). The budget therefore
//! means "stop once the tableau walk plus interleaved tightening has run
//! `secs` seconds," and `done: ... secs=` is OP-WALK WALL, not full root-build
//! wall and not phase compute. The phase's own compute is
//! `work_secs=` (summed over completed passes) and nothing else. Reading
//! `secs=` as phase cost produced a banked, campaign-steering falsehood —
//! "#backward-interm overruns its armed budget 2.11x in 195/195 tinyimagenet
//! logs and eats ~45% of the lane" — retracted after direct measurement
//! 2026-08-24 (idx_5451, official 100 s): the phase's compute was 2.83 s of a
//! 48.9 s walk interval, one 256-column pass was 0.64 s, and `exhausted=true`
//! meant the FORWARD OP WALK had spent the shared clock, cutting the phase at
//! trunk layer 2 of ~10 — starved, not greedy. The dose-response that
//! followed (docs/BI_DOSE_RESPONSE_5185_2026-08-24.md) is measured in these
//! corrected units.

use std::time::Instant;

use ndarray::{Array2, Axis};
use ny_core::{NyError, Result};

use super::bi_gpu::{self, BiGpu, BiGpuScope, PairMark};
use super::engine::{BackwardEngine, LaneDir, Seed};
use super::gpu_conv::GpuConvRoute;
use super::net::{TwinNet, TwinOp};
use super::root::{ClipRows, LayerGates, RootGates};
use super::rounding::RoundMode;

/// Tuning for one build's backward-intermediate phase.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BiCfg {
    /// Wall-clock budget in seconds (before the 40%-of-deadline cap).
    pub secs: f64,
    /// Columns per prefix pass (memory / batching grain).
    pub chunk: usize,
    /// Target seconds per prefix pass. Zero preserves fixed-width chunking.
    pub chunk_secs: f64,
    /// Max unstable neurons re-derived per layer (widest first).
    pub topk: usize,
    /// #bi-yield: stop the phase once it stops PAYING.
    ///
    /// Evaluate the marginal tightening rate every `yield_window` completed
    /// LAYERS; if a layer's rate falls below `yield_ratio` x the best layer seen,
    /// declare the phase exhausted and hand the rest of the budget to the tree
    /// search. `0` disables the check and is byte-identical to the fixed
    /// wall-budget behaviour.
    ///
    /// # Why LAYERS and not chunks
    ///
    /// Measured 2026-09-16 on cifar100 resnet_large: the phase completes only
    /// FOUR layers inside its whole budget (1024, 301, 219 and 1 columns), a
    /// handful of chunks each. A chunk-keyed window is still establishing its
    /// baseline when the budget exhausts, so the check never fires — armed and
    /// inert, which reads as a negative result and is not one. The layer is this
    /// phase's real unit of work and its rate signal is unambiguous: the same
    /// run measured 161, 36 and 22 tightened/s across its three working layers.
    pub yield_window: usize,
    /// Fraction of the best observed window rate below which the phase stops.
    pub yield_ratio: f64,
    /// #interm-alpha: how many PER-START-NODE alpha candidates each layer
    /// tries beyond the shared heuristic. `0` (default) is byte-identical to
    /// the pre-#interm-alpha phase; `k` costs `k` extra Lower+Upper prefix
    /// pass pairs per chunk and can only shrink the published box.
    pub interm_alpha: usize,
}

/// Candidate wall a layer must have charged before `#ia-yield` will price it.
/// Below this the layer ran no candidates at all, so its rate is an artifact of
/// the divide-by-zero floor rather than a measurement.
const IA_YIELD_MIN_CHARGE_SECS: f64 = 1e-3;

/// #interm-alpha: how many distinct per-start-node alpha policies exist.
/// A dose above this is clamped to it (every policy beyond the last would
/// re-run an identical pass and spend budget for a guaranteed zero).
pub(crate) const INTERM_ALPHA_POLICIES: usize = 5;

/// #bi-window: target columns per spatial TILE when the phase orders its
/// selected neurons tile-major (`window::tile_order`). A chunk never straddles
/// two tiles, so its receptive-field crop is one compact rectangle. Smaller
/// tiles crop harder but pass fewer columns per certified pass; 64 keeps the
/// engine's length-`R` inner loops wide while the crop at the input of a
/// CIFAR CNN7 trunk shrinks ~3-10x per layer. Scheduling only: the tile size
/// changes which columns share a pass, never what any column's pass computes.
pub(crate) const WINDOW_TARGET_COLS: usize = 64;

/// Closed execution receipt for one root build's frozen backward-intermediate
/// policy.  Configuration fields identify the treatment; counters prove which
/// values actually constrained live work.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackwardIntermExecutionEvidence {
    pub enabled: bool,
    /// MEASURED WORK of this phase — the summed chunk wall — as distinct from
    /// `configured_secs` (its BUDGET).
    ///
    /// ⚠ THIS IS `work_secs`, NOT the `done: secs=` figure. This module's own
    /// field docs warn that `done: secs=` is TABLEAU-OP-WALK wall, not phase
    /// work: the phase's `t0` spans the whole op walk it is interleaved with, so
    /// `t0.elapsed()` OVERSTATES what the phase costs. Pricing a half-dose round
    /// off that number would over-credit the saving. Used by `#rebake`'s round-1
    /// affordability pricing. Zero when the phase never ran.
    pub measured_work_secs: f64,
    pub configured_secs: f64,
    pub configured_chunk: usize,
    pub configured_chunk_secs: f64,
    pub effective_chunk_secs: f64,
    pub configured_topk: usize,
    pub eligible_layers: usize,
    pub suppressed_setup_layers: usize,
    pub layers_started: usize,
    pub chunks_completed: usize,
    pub columns_completed: usize,
    pub deadline_capped: bool,
    pub budget_exhausted: bool,
    pub chunk_secs_capped: bool,
    pub chunk_counterfactual_chunks: usize,
    pub chunk_ceiling_binding_chunks: usize,
    pub fixed_width_chunks: usize,
    pub adaptive_width_changes: usize,
    pub topk_binding_layers: usize,
    pub policy_mismatch: bool,
}

impl BackwardIntermExecutionEvidence {
    pub(crate) fn from_policy(policy: super::BackwardIntermRuntimePolicy) -> Self {
        Self {
            enabled: policy.enabled,
            measured_work_secs: 0.0,
            configured_secs: policy.secs,
            configured_chunk: policy.chunk,
            configured_chunk_secs: policy.chunk_secs,
            effective_chunk_secs: policy.chunk_secs,
            configured_topk: policy.topk,
            eligible_layers: 0,
            suppressed_setup_layers: 0,
            layers_started: 0,
            chunks_completed: 0,
            columns_completed: 0,
            deadline_capped: false,
            budget_exhausted: false,
            chunk_secs_capped: false,
            chunk_counterfactual_chunks: 0,
            chunk_ceiling_binding_chunks: 0,
            fixed_width_chunks: 0,
            adaptive_width_changes: 0,
            topk_binding_layers: 0,
            policy_mismatch: false,
        }
    }

    /// Stable phase outcome carried by the CLI event grammar.
    #[must_use]
    pub fn outcome(self) -> &'static str {
        if !self.enabled {
            "disabled"
        } else if self.budget_exhausted {
            "exhausted"
        } else {
            "completed"
        }
    }
}

impl Default for BackwardIntermExecutionEvidence {
    fn default() -> Self {
        Self::from_policy(super::BackwardIntermRuntimePolicy::default())
    }
}

/// Width of the next independent-column prefix pass.
///
/// A zero target is the historical fixed grain. An adaptive layer begins with
/// one probe column, then chooses a width whose linearized cost fits the target
/// and never exceeds the fixed grain's memory ceiling.
fn next_chunk_width(
    target_secs: f64,
    seconds_per_col: Option<f64>,
    chunk_ceiling: usize,
    remaining_cols: usize,
) -> usize {
    let chunk_ceiling = chunk_ceiling.max(1);
    let requested = if target_secs == 0.0 {
        chunk_ceiling
    } else if let Some(rate) = seconds_per_col {
        ((target_secs / rate).floor() as usize).clamp(1, chunk_ceiling)
    } else {
        1
    };
    requested.min(remaining_cols)
}

/// Conservative wall estimate for deadline admission of the next chunk.
///
/// `rate * width` prices a post-probe width jump instead of reusing the
/// one-column probe wall. The `max(last)` term retains the measured fixed
/// overhead when the final chunk is narrower; without it, scaling an affine
/// pass cost down by column count would under-price that tail.
fn predicted_chunk_secs(
    last_chunk_secs: f64,
    seconds_per_col: Option<f64>,
    width: usize,
) -> Option<f64> {
    if last_chunk_secs <= 0.0 {
        return None;
    }
    Some(seconds_per_col.map_or(last_chunk_secs, |rate| {
        last_chunk_secs.max(rate * width as f64)
    }))
}

/// Whether the configured chunk ceiling changes this completed chunk's width
/// relative to the declaration default under the same measured adaptive rate.
fn chunk_width_differs_from_default(
    target_secs: f64,
    seconds_per_col: Option<f64>,
    configured_chunk: usize,
    remaining_cols: usize,
) -> bool {
    next_chunk_width(
        target_secs,
        seconds_per_col,
        configured_chunk,
        remaining_cols,
    ) != next_chunk_width(
        target_secs,
        seconds_per_col,
        super::DEFAULT_BACKWARD_INTERM_CHUNK,
        remaining_cols,
    )
}

/// Whether this configured ceiling reduced the live width that would have run
/// with no batching ceiling. Unlike fixed-mode work alone, this proves the
/// configured value constrained a completed chunk.
fn chunk_ceiling_binds_width(
    target_secs: f64,
    seconds_per_col: Option<f64>,
    configured_chunk: usize,
    remaining_cols: usize,
) -> bool {
    next_chunk_width(
        target_secs,
        seconds_per_col,
        configured_chunk,
        remaining_cols,
    ) < next_chunk_width(target_secs, seconds_per_col, remaining_cols, remaining_cols)
}

/// #interm-alpha: the PREFIX gate override one start node uses for one
/// candidate policy.
///
/// THE AXIS. `backward_interm` bounds trunk relu `prefix_len`'s input box by a
/// backward pass over the frozen prefix — alpha-beta-CROWN's intermediate-bound
/// computation. abc holds a SEPARATE optimizable alpha set for every
/// (relu layer, downstream START NODE) pair — `alpha[relu][start_node]` of
/// shape `[2, spec, batch, *relu_shape]`, verified against
/// `references/alpha-beta-CROWN/auto_LiRPA/auto_LiRPA/operators/relu.py:155`
/// and `backward_bound.py:874-944`, and abc's initial 20-iteration alpha-CROWN
/// OPTIMIZES the intermediate set (`fix_interm_bounds=True` is inert there
/// because `interm_bounds` is `None`) — so each intermediate layer's bound is
/// computed under slopes chosen FOR THAT LAYER. ny reuses the one shared
/// `(layer, neuron)` heuristic alpha for every start node, which is the third
/// axis this function opens: the same prefix, re-relaxed per start node.
///
/// THE CANDIDATES. Both are closed-form and defensible, not random probes:
/// * policy 0, `alpha = s`: the lower line PARALLEL to the certified upper
///   chord — CROWN's other classic adaptive choice; the shipped heuristic
///   `[u >= -l]` snaps to a vertex, optimal only at the extremes.
/// * policy 1, the FLIPPED binary: whichever of `{0, 1}` the area heuristic
///   did NOT pick — the heuristic is blind to the coefficients actually
///   reaching the neuron from THIS start node.
///
/// SOUNDNESS. `alpha * z <= relu(z)` holds pointwise on all of `R` for ANY
/// `alpha in [0, 1]`; only UNSTABLE neurons are touched; the upper chord
/// `(s, c)` is a function of `(l, u)` only and is NOT moved; the error-carry
/// constant `ms = max(alpha, s)` is re-derived from the CANDIDATE alpha here
/// (a stale `ms` under a raised alpha would UNDER-scale the carried error —
/// a false-`unsat` generator).
///
/// MEMORY. Four f64 vectors of layer width per prefix layer, one policy at a
/// time: ~1.8 MB on CIFAR100_resnet_medium, ~9.2 MB on resnet_large.
/// Deliberately NOT per-(neuron, column): at the phase's 256-column chunk that
/// is ~134 MB for one 65,536-wide layer and over a gigabyte across the prefix,
/// on a unified-memory host where an over-admission is a global OOM.
///
/// RETURNS THE COUNT OF ALPHA ACTUALLY MOVED, and that count is telemetry, not
/// bookkeeping. `armed=1` proves the BINARY carries the lever; it does not
/// prove the CONFIG gave the lever anything to work on (a shipped
/// constrained-zonotope lane ran as plain DeepZ for want of three literal `0`
/// caps, and every null through it was VOID). A dose whose `alpha_moved` is
/// flat across doses makes a flat dose-response VOID, not negative.
fn start_node_gates(
    root: &RootGates,
    prefix_len: usize,
    policy: usize,
) -> Option<(super::engine::DomainGates, usize)> {
    use super::engine::{DomainGates, GateVecs};
    let mut dg = DomainGates::default();
    let mut moved = 0usize;
    for li in 0..prefix_len.min(root.layers.len()) {
        let layer = &root.layers[li];
        if layer.unst.is_empty() {
            continue;
        }
        let mut alpha = layer.alpha.clone();
        for &j in &layer.unst {
            // Every candidate is a slope in [0, 1], so `alpha * z <= relu(z)`
            // holds pointwise for all of them and each produces a VALID
            // enclosure; the caller intersects them shrink-only, so adding
            // candidates is monotonically tighter and can only cost wall
            // (charged as `ia_secs`, polled by `out_of_time`).
            // `s = u / (u - l)` with `l < 0 < u` is nominally interior to
            // [0, 1], but `repair_upper_lines` can push it to the boundary in
            // outward mode, and then `(s + 1) / 2` would exceed 1 — so the
            // `clamp(0.0, 1.0)` below is LOAD-BEARING, not defensive tidiness.
            let cand = match policy {
                // Parallel to the upper chord.
                0 => layer.s[j],
                // The vertex the area heuristic rejected.
                1 => {
                    if layer.alpha[j] >= 0.5 {
                        0.0
                    } else {
                        1.0
                    }
                }
                // The three interpolations the two vertices and the chord
                // leave unexplored. abc OPTIMIZES this slope per start node
                // and gains a median +0.163 at the root where ny's two fixed
                // candidates gain +0.0025; a short defensible grid is the
                // cheap stand-in for that ascent.
                2 => 0.5,
                // Between the chord-parallel line and fully-active.
                3 => layer.s[j].midpoint(1.0),
                // Between fully-inactive and the chord-parallel line.
                _ => layer.s[j] * 0.5,
            };
            let cand = cand.clamp(0.0, 1.0);
            if cand != alpha[j] {
                moved += 1;
            }
            alpha[j] = cand;
        }
        // `ms` re-derived from the CANDIDATE alpha, by the same
        // `max(alpha, s)` formula the root build uses.
        let ms: Vec<f64> = alpha.iter().zip(&layer.s).map(|(a, s)| a.max(*s)).collect();
        dg.layers.insert(
            li,
            GateVecs {
                alpha,
                s: layer.s.clone(),
                c: layer.c.clone(),
                ms,
            },
        );
    }
    // A candidate identical to the shipped gates would spend a pass pair for a
    // guaranteed zero. `moved == 0` is also the config-starvation signature.
    (moved > 0).then_some((dg, moved))
}

/// Per-build state: budget clock + telemetry accumulators.
pub(crate) struct BiState {
    enabled: bool,
    cfg: BiCfg,
    t0: Instant,
    /// Effective wall budget (cfg.secs capped to 40% of the remaining
    /// deadline at construction; the tree search is never starved).
    budget: f64,
    /// Effective adaptive target after the 10%-of-budget cap. Zero keeps the
    /// historical fixed-width mode.
    effective_chunk_secs: f64,
    chunk_secs_capped: bool,
    deadline: Option<Instant>,
    exhausted: bool,
    /// Totals across layers (printed per layer; summed for the final line).
    tightened_total: usize,
    stabilized_total: usize,
    crossed_total: usize,
    pass_errors: usize,
    /// Prefix-pass compute only; unlike `t0`, excludes the forward root walk.
    work_secs: f64,
    /// #bi-chunk-predict: measured wall of the last completed chunk, and how
    /// many chunks the predictor refused to start.
    ///
    /// `out_of_time` is polled BETWEEN chunks and never inside one, so a chunk
    /// that starts always completes. The last completed chunk's measured wall
    /// prices the next admission and bounds predictable overshoot. Historical
    /// claims derived from `done: secs=` are intentionally absent here: that
    /// field is tableau-op-walk wall, not phase work (see THE TWO CLOCKS above).
    last_chunk_secs: f64,
    chunks_declined: usize,
    /// #bi-yield accounting: the window baseline and the best rate seen.
    yield_last_tightened: usize,
    yield_last_secs: f64,
    yield_last_chunks: usize,
    /// Layers whose work has completed — the #bi-yield window unit.
    yield_layers_done: usize,
    yield_best_rate: f64,
    /// Set when the marginal-yield check, not the wall budget, ended the phase.
    yield_stopped: bool,
    /// `#ia-yield`: marginal-value governor for the `#interm-alpha` candidate
    /// loop ONLY. Separate from `#bi-yield` above, which ends the whole phase.
    ia_yield: super::marginal::MarginalYield,
    /// Set once `#ia-yield` has fired: later layers run the baseline pass but
    /// skip the per-start-node candidates.
    ia_yield_stopped: bool,
    /// `self.ia_secs` as of the previous layer's observation, so a layer's own
    /// candidate wall can be differenced out of the cumulative charge.
    ia_yield_last_secs: f64,
    /// Layer index at which `#ia-yield` fired, for the lever's observable.
    ia_yield_fired_layer: Option<usize>,
    /// #router-r3: the rate the firing layer observed (its ledger entry).
    ia_yield_fired_rate: f64,
    /// Causal execution counters for the closed T1 receipt.
    eligible_layers: usize,
    suppressed_setup_layers: usize,
    layers_started: usize,
    chunks_completed: usize,
    columns_completed: usize,
    deadline_capped: bool,
    chunk_counterfactual_chunks: usize,
    chunk_ceiling_binding_chunks: usize,
    fixed_width_chunks: usize,
    adaptive_width_changes: usize,
    topk_binding_layers: usize,
    /// #interm-alpha marginal telemetry: contributions of the per-start-node
    /// candidates ALONE (a neuron already tightened/stabilized by the shared
    /// heuristic pass is not re-counted), pass-pair count, wall, and the
    /// per-POLICY count of alphas actually put in play (anti-starvation).
    ia_tightened_total: usize,
    ia_stabilized_total: usize,
    ia_pass_pairs: usize,
    ia_pass_errors: usize,
    ia_secs: f64,
    ia_alpha_moved: [usize; INTERM_ALPHA_POLICIES],
    /// #mr-errlane: the certified error lane of the root being built, handed
    /// to this phase's prefix-only scratch root (and so to its engine). The
    /// prefix passes are the widest the lane runs (`cols` up to the chunk).
    err_lane: super::ErrLane,
    /// #bi-window: run each chunk's prefix pair on a RECEPTIVE-FIELD CROP of
    /// the prefix net (`window` module) instead of the dense full-tensor walk.
    /// Copied from the frozen policy (`margin_row.bi_window`); `false` — the
    /// declaration default — never enters the crop code, so every pass is the
    /// historical dense pass, byte-identical.
    window: bool,
    /// #bi-window telemetry: cropped pass PAIRS completed; dense fallbacks (a
    /// layer or chunk the crop cannot express — e.g. a residual `Add` in the
    /// prefix); and the walked rows of the cropped pairs against the rows the
    /// same pairs walk dense. `row_ratio = window / dense` is the work the
    /// crop removed. An armed run whose `pairs=0` is VOID for this lever.
    win_pairs: usize,
    win_fallbacks: usize,
    win_rows_window: usize,
    win_rows_dense: usize,
    /// #bi-gpu: this phase's scoped device route and its per-pair accounting
    /// (`bi_gpu` module). Built only when the frozen policy arms
    /// `margin_row.bi_gpu` on an enabled phase; `None` — the declaration
    /// default — builds no route, so every pass is the historical CPU pass,
    /// byte-identical. Its `[bi-gpu] done` receipt prints when the state drops,
    /// on every exit of the build.
    bi_gpu: Option<BiGpu>,
    /// #bi-patch: run each chunk's prefix pairs as PER-COLUMN RECEPTIVE-FIELD
    /// walks (`patch` module). Copied from the frozen policy
    /// (`margin_row.bi_patch`, only when the phase is enabled); `None` — the
    /// declaration default — never builds a plan, so every pass is the
    /// historical pass, byte-identical. Takes precedence over `window` on
    /// every layer it can plan.
    patch: Option<super::patch::PatchCfg>,
    /// #bi-patch telemetry: patch pairs completed (baseline and candidate),
    /// their pass columns (`2 x` the requested columns: Lower and Upper are
    /// fused), plan or chunk refusals that fell back, pairs whose walk
    /// switched to the dense tail, signed conv MACs patch vs dense, walked
    /// rows per column patch vs dense, and the pairs' wall. An armed run whose
    /// `pairs=0` is VOID for this lever.
    patch_pairs: usize,
    patch_fused_cols: usize,
    patch_fallbacks: usize,
    patch_converted: usize,
    patch_macs_patch: f64,
    patch_macs_dense: f64,
    patch_rows_patch: usize,
    patch_rows_dense: usize,
    patch_secs: f64,
}

/// Telemetry from one incremental epoch refresh.
///
/// The counts are evidence-only except `tightened`, which the BaB caller uses
/// to refuse a zero-progress nested run. No statistic can establish a closure
/// or alter a certified numeric bound.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ParentRefreshStats {
    /// Exact split gates baked into the cloned parent frame.
    pub(crate) baked: usize,
    /// Downstream ReLU layers for which at least one column was requested.
    pub(crate) layers: usize,
    /// Preactivation columns re-bounded by the certified prefix engine.
    pub(crate) columns: usize,
    /// Re-bounded columns whose parent interval shrank.
    pub(crate) tightened: usize,
    /// Tightened columns that became sign-stable.
    pub(crate) stabilized: usize,
    /// Wall time for the complete clone + refresh.
    pub(crate) elapsed_secs: f64,
}

/// Intersect one parent/fresh interval or refuse the entire incremental frame.
///
/// A crossing cannot be consumed as an emptiness verdict here: this path owns
/// no independent non-emptiness proof. It is instead a numeric/invariant defect
/// signal. Returning an error discards the cloned frame — including any earlier
/// layer updates — and leaves ordinary certified expansion in charge.
fn intersect_parent_refresh(
    parent_l: f64,
    parent_u: f64,
    fresh_l: f64,
    fresh_u: f64,
    layer: usize,
    neuron: usize,
) -> Result<(f64, f64)> {
    let lower = parent_l.max(fresh_l);
    let upper = parent_u.min(fresh_u);
    if lower > upper {
        return Err(NyError::NumericalInstability(format!(
            "margin_row: parent/fresh epoch interval crossed at ({layer},{neuron}): \
             parent=[{parent_l},{parent_u}] fresh=[{fresh_l},{fresh_u}]"
        )));
    }
    Ok((lower, upper))
}

/// Incrementally refresh an epoch from its already-tight parent frame.
///
/// A full [`RootGates::build_retaining`](super::root::RootGates::build_retaining)
/// starts again at `root.lo`/`root.hi` and carries an input-relative tableau
/// through every trunk op.  That is the measured ~7 s operation on
/// CIFAR100-resnet-large.  This path instead:
///
/// 1. clones the parent's published intervals and certified gates;
/// 2. bakes the domain's split ReLUs as exact active/inactive lines; and
/// 3. re-bounds only `topk` downstream unstable preactivations per layer with
///    [`BackwardEngine::run_prefix`], intersecting each result with the parent
///    interval before re-deriving that neuron's gate.
///
/// The expensive dimension is therefore selected columns, not the full
/// input-relative tableau.  Earlier refreshed layers immediately feed later
/// prefix passes, so the tightening compounds in execution order.
///
/// # Soundness
///
/// Let `D` be the parent epoch region intersected with `splits`.  Parent
/// intervals remain valid on `D` because `D` is a subset.  A baked active
/// (`identity`) or inactive (`zero`) ReLU is exact on its corresponding split
/// halfspace.  `run_prefix` then applies only certified relaxations that are
/// valid on `D`, producing an affine enclosure there; concretizing that affine
/// expression over the *larger* declared input box can only loosen its extrema.
/// Thus both the parent interval and the fresh prefix interval enclose the true
/// preactivation range on `D`, and their intersection is sound.  Gates are
/// rebuilt through `gates_from_box` + `repair_upper_lines`, exactly as in the
/// root build.
///
/// Crossed intersections never certify emptiness here. They refuse the entire
/// cloned frame as a numeric/invariant defect. Malformed splits, non-finite
/// passes, crossings, or deadline expiry return an error; the BaB caller fails
/// open to ordinary expansion.
///
/// `chunk` changes only batching/memory.  `topk == 0`, parity mode, or an empty
/// split set is rejected so the default path cannot enter this implementation
/// accidentally.
pub(crate) fn refresh_parent_frame(
    net: &TwinNet,
    parent: &RootGates,
    splits: &[(usize, usize, i8)],
    topk: usize,
    chunk: usize,
    deadline: Option<Instant>,
) -> Result<(RootGates, ParentRefreshStats)> {
    if !parent.mode.outward() {
        return Err(NyError::InvalidSpec(
            "margin_row: parent-frame epoch refresh requires outward mode".into(),
        ));
    }
    if splits.is_empty() || topk == 0 {
        return Err(NyError::InvalidSpec(
            "margin_row: parent-frame epoch refresh needs splits and topk > 0".into(),
        ));
    }
    // Check before cloning or baking, including requests with no downstream layer.
    if deadline.is_some_and(|dl| Instant::now() >= dl) {
        return Err(NyError::DeadlineExceeded(
            "margin_row parent-frame epoch refresh deadline".into(),
        ));
    }
    let t0 = Instant::now();
    // Clone only the compact certified frame. `clip_rows` can be hundreds of
    // MiB and is indexed by POSITION in `unst`; copying it just to discard it
    // would erase the intended amortization, while retaining it after the
    // baked/stabilized removals below would silently re-point split rows.
    let mut child = RootGates {
        mode: parent.mode,
        mid: parent.mid.clone(),
        rad: parent.rad.clone(),
        xabs: parent.xabs.clone(),
        layers: parent
            .layers
            .iter()
            .map(|layer| LayerGates {
                op: layer.op,
                n: layer.n,
                l: layer.l.clone(),
                u: layer.u.clone(),
                alpha: layer.alpha.clone(),
                s: layer.s.clone(),
                c: layer.c.clone(),
                ms: layer.ms.clone(),
                unst: layer.unst.clone(),
                clip_rows: None,
            })
            .collect(),
        lo: parent.lo.clone(),
        hi: parent.hi.clone(),
        backward_interm_execution: parent.backward_interm_execution,
        root_precision_policy: parent.root_precision_policy,
        root_precision_execution: parent.root_precision_execution,
        // #spec-alpha fires ONCE PER INSTANCE at the root direct pass, never
        // per epoch: an epoch re-derives gates through `gates_from_box` and a
        // second firing would re-pay the pass for a frame the axis was not
        // measured against. Deliberately NOT inherited from `parent`.
        spec_alpha_policy: None,
        // #head-alpha IS inherited: the trunk alpha above is carried from the
        // parent verbatim, and the head slopes are valid on any sub-box (they
        // are consumed only where a head neuron is still unstable).
        head_alpha: parent.head_alpha.clone(),
        // #root-eval-reuse: the child's gates differ; it inherits nothing.
        root_eval_memo: super::bab::RootEvalMemo::default(),
        head_group_source: None,
    };

    let mut baked_by_layer: std::collections::BTreeMap<
        usize,
        std::collections::BTreeMap<usize, i8>,
    > = std::collections::BTreeMap::new();
    let mut earliest = usize::MAX;
    for &(li, idx, dir) in splits {
        let Some(layer) = child.layers.get(li) else {
            return Err(NyError::InvalidSpec(format!(
                "margin_row: parent-frame split layer {li} out of range {}",
                child.layers.len()
            )));
        };
        if idx >= layer.n || !layer.unst.contains(&idx) || !matches!(dir, -1 | 1) {
            return Err(NyError::InvalidSpec(format!(
                "margin_row: malformed parent-frame split ({li},{idx},{dir})"
            )));
        }
        if baked_by_layer
            .entry(li)
            .or_default()
            .insert(idx, dir)
            .is_some()
        {
            return Err(NyError::InvalidSpec(format!(
                "margin_row: duplicate parent-frame split ({li},{idx})"
            )));
        }
        earliest = earliest.min(li);
    }

    // Bake all inherited split constraints before the first prefix pass.  The
    // sign-clipped preactivation interval is also valid on the child region and
    // gives the nested frame an honest published box for the fixed neuron.
    for (&li, by_neuron) in &baked_by_layer {
        let layer = &mut child.layers[li];
        for (&idx, &dir) in by_neuron {
            if dir > 0 {
                layer.l[idx] = layer.l[idx].max(0.0);
                layer.alpha[idx] = 1.0;
                layer.s[idx] = 1.0;
                layer.c[idx] = 0.0;
                layer.ms[idx] = 1.0;
            } else {
                layer.u[idx] = layer.u[idx].min(0.0);
                layer.alpha[idx] = 0.0;
                layer.s[idx] = 0.0;
                layer.c[idx] = 0.0;
                layer.ms[idx] = 0.0;
            }
            if layer.l[idx] > layer.u[idx] {
                // This would establish an empty split region, but this speed
                // path owns no emptiness verdict. Refuse and let the ordinary
                // certified domain path handle it.
                return Err(NyError::NumericalInstability(format!(
                    "margin_row: parent-frame split interval crossed at ({li},{idx})"
                )));
            }
        }
        layer.unst.retain(|idx| !by_neuron.contains_key(idx));
    }

    let mut stats = ParentRefreshStats {
        baked: splits.len(),
        ..ParentRefreshStats::default()
    };
    let chunk = chunk.max(1);
    // A split cannot change another preactivation in the SAME layer. Start at
    // the first downstream layer; every refreshed gate then compounds into the
    // next layer's prefix pass.
    for li in earliest.saturating_add(1)..child.layers.len() {
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            return Err(NyError::DeadlineExceeded(
                "margin_row parent-frame epoch refresh deadline".into(),
            ));
        }
        let layer = &child.layers[li];
        let mut selected: Vec<(f64, usize)> = layer
            .unst
            .iter()
            .map(|&idx| (layer.u[idx] - layer.l[idx], idx))
            .filter(|(width, _)| width.is_finite() && *width > 0.0)
            .collect();
        selected.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        selected.truncate(topk);
        let mut columns: Vec<usize> = selected.into_iter().map(|(_, idx)| idx).collect();
        columns.sort_unstable();
        if columns.is_empty() {
            continue;
        }
        stats.layers += 1;
        let relu_op = layer.op;
        let n_pre = layer.n;
        let mut fresh = Vec::with_capacity(columns.len());
        for cols in columns.chunks(chunk) {
            if deadline.is_some_and(|dl| Instant::now() >= dl) {
                return Err(NyError::DeadlineExceeded(
                    "margin_row parent-frame epoch refresh deadline".into(),
                ));
            }
            let mut s = Array2::<f64>::zeros((n_pre, cols.len()));
            for (column, &idx) in cols.iter().enumerate() {
                s[[idx, column]] = 1.0;
            }
            let seed = Seed { s, e: None };
            // Keep the immutable engine borrow inside the chunk. The layer is
            // mutated only after all of its prefix rows have been certified.
            let (lb, ub) = {
                let eng = BackwardEngine::new(net, &child);
                let lower = eng.run_prefix(&seed, relu_op, LaneDir::Lower)?;
                let upper = eng.run_prefix(&seed, relu_op, LaneDir::Upper)?;
                (eng.concretize_lower(&lower), eng.concretize_upper(&upper))
            };
            for (column, &idx) in cols.iter().enumerate() {
                if !(lb[column].is_finite() && ub[column].is_finite()) {
                    return Err(NyError::NumericalInstability(format!(
                        "margin_row: non-finite parent-frame prefix bound at ({li},{idx})"
                    )));
                }
                fresh.push((idx, lb[column], ub[column]));
            }
            stats.columns += cols.len();
        }

        let layer = &mut child.layers[li];
        let mut touched = Vec::new();
        for (idx, lb, ub) in fresh {
            let (nl, nu) = intersect_parent_refresh(layer.l[idx], layer.u[idx], lb, ub, li, idx)?;
            if nl > layer.l[idx] || nu < layer.u[idx] {
                let was_unstable = layer.l[idx] < 0.0 && layer.u[idx] > 0.0;
                layer.l[idx] = nl;
                layer.u[idx] = nu;
                stats.tightened += 1;
                if was_unstable && (nl >= 0.0 || nu <= 0.0) {
                    stats.stabilized += 1;
                }
                touched.push(idx);
            }
        }
        if touched.is_empty() {
            continue;
        }
        let (alpha, mut s, mut c) = super::root::gates_from_box(&layer.l, &layer.u);
        super::root::repair_upper_lines(&layer.l, &layer.u, &mut s, &mut c);
        for &idx in &touched {
            layer.alpha[idx] = alpha[idx];
            layer.s[idx] = s[idx];
            layer.c[idx] = c[idx];
            layer.ms[idx] = alpha[idx].max(s[idx]);
        }
        let baked = baked_by_layer.get(&li);
        layer.unst = (0..layer.n)
            .filter(|&idx| {
                layer.l[idx] < 0.0
                    && layer.u[idx] > 0.0
                    && !baked.is_some_and(|set| set.contains_key(&idx))
            })
            .collect();
    }
    stats.elapsed_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[epoch-parent] refreshed: baked={} layers={} columns={} tightened={} \
         stabilized={} secs={:.3}",
        stats.baked,
        stats.layers,
        stats.columns,
        stats.tightened,
        stats.stabilized,
        stats.elapsed_secs
    );
    // The original deadline also governs complete zero-work and final-layer paths.
    if deadline.is_some_and(|dl| Instant::now() >= dl) {
        return Err(NyError::DeadlineExceeded(
            "margin_row parent-frame epoch refresh deadline".into(),
        ));
    }
    Ok((child, stats))
}

/// #joint-refresh (cct plan P5 step 2): telemetry of ONE backward-only
/// intermediate refresh of a root frame. Evidence only — no count here feeds a
/// bound; `tightened == 0` is the driver's fixed-point signal.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct JointRefreshStats {
    /// Trunk relu layers that had at least one column re-bounded.
    pub(crate) layers: usize,
    /// Pre-activation columns re-bounded (each is one Lower+Upper pair column).
    pub(crate) columns: usize,
    /// Columns whose published box strictly shrank.
    pub(crate) tightened: usize,
    /// Tightened columns that became sign-stable (removed from `unst`).
    pub(crate) stabilized: usize,
    /// Columns skipped because a pass returned a non-finite endpoint.
    pub(crate) nonfinite: usize,
    /// Cropped (#bi-window) pass pairs, dense pass pairs, and crop refusals.
    pub(crate) windowed_pairs: usize,
    pub(crate) dense_pairs: usize,
    pub(crate) window_fallbacks: usize,
    /// Walked rows of the cropped pairs vs the rows they would walk dense.
    pub(crate) rows_window: usize,
    pub(crate) rows_dense: usize,
    /// #bi-patch: patch pass pairs, plan/chunk refusals that fell back, and
    /// the walked rows per column of the patch pairs vs the dense walk.
    pub(crate) patch_pairs: usize,
    pub(crate) patch_fallbacks: usize,
    pub(crate) rows_patch: usize,
    pub(crate) rows_patch_dense: usize,
    /// The refresh stopped at its wall slice (the partial frame is published —
    /// every applied intersection was certified on its own).
    pub(crate) deadline_cut: bool,
    /// Wall seconds of the whole refresh.
    pub(crate) secs: f64,
}

/// One certified prefix Lower+Upper pair for `cols` of trunk relu op
/// `relu_op` under `root`'s CURRENT gates: a per-column receptive-field patch
/// pair (#bi-patch) when `plan` is given and can express the chunk, else
/// cropped (#bi-window) when `shapes` is given and the crop can express the
/// chunk, dense otherwise.
///
/// `gpu` (#bi-gpu): the refresh's device route. `None` — always, when dark —
/// runs exactly the historical pair; an RSS-refused pair runs wholly on the
/// CPU kernel.
#[allow(clippy::too_many_arguments)]
fn refresh_prefix_pair(
    net: &TwinNet,
    root: &RootGates,
    relu_op: usize,
    n_pre: usize,
    cols: &[usize],
    shapes: Option<&[window::Shape]>,
    plan: Option<&super::patch::PatchPlan>,
    stats: &mut JointRefreshStats,
    gpu: Option<&mut BiGpu>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let Some(gpu) = gpu else {
        return refresh_prefix_pair_on(net, root, relu_op, n_pre, cols, shapes, plan, stats, None);
    };
    let Some(mark) = gpu.open_pair() else {
        // RSS-refused: the whole pair runs on the CPU kernel.
        let cpu = GpuConvRoute::Off;
        return refresh_prefix_pair_on(
            net,
            root,
            relu_op,
            n_pre,
            cols,
            shapes,
            plan,
            stats,
            Some(&cpu),
        );
    };
    let route = gpu.route();
    let out = refresh_prefix_pair_on(
        net,
        root,
        relu_op,
        n_pre,
        cols,
        shapes,
        plan,
        stats,
        Some(&route),
    );
    gpu.close_pair(mark);
    out
}

/// [`refresh_prefix_pair`]'s body on an explicit conv route (`None`: the
/// engine's own, i.e. the historical pair).
#[allow(clippy::too_many_arguments)]
fn refresh_prefix_pair_on(
    net: &TwinNet,
    root: &RootGates,
    relu_op: usize,
    n_pre: usize,
    cols: &[usize],
    shapes: Option<&[window::Shape]>,
    plan: Option<&super::patch::PatchPlan>,
    stats: &mut JointRefreshStats,
    route: Option<&GpuConvRoute>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    // #bi-patch: the per-column receptive-field pair first; dark, `plan` is
    // `None` and this is never attempted.
    if let Some(plan) = plan {
        match plan.prefix_bounds(net, root, cols, None, route) {
            Some(Ok(pair)) => {
                stats.patch_pairs += 1;
                stats.rows_patch = stats.rows_patch.saturating_add(pair.rows_patch);
                stats.rows_patch_dense = stats.rows_patch_dense.saturating_add(pair.rows_dense);
                return Ok((pair.lb, pair.ub));
            }
            Some(Err(error)) => return Err(error),
            None => stats.patch_fallbacks += 1,
        }
    }
    if let Some(shapes) = shapes {
        match window::prefix_bounds(net, root, relu_op, shapes, cols, None, route) {
            Some(Ok(pair)) => {
                stats.windowed_pairs += 1;
                stats.rows_window = stats.rows_window.saturating_add(pair.rows_window);
                stats.rows_dense = stats.rows_dense.saturating_add(pair.rows_dense);
                return Ok((pair.lb, pair.ub));
            }
            Some(Err(error)) => return Err(error),
            None => stats.window_fallbacks += 1,
        }
    }
    let mut s = Array2::<f64>::zeros((n_pre, cols.len()));
    for (column, &j) in cols.iter().enumerate() {
        s[[j, column]] = 1.0;
    }
    // Identity seed is exact: the engine zero-fills its error lane.
    let seed = Seed { s, e: None };
    let eng = bi_gpu::engine(net, root, route);
    let lower = eng.run_prefix(&seed, relu_op, LaneDir::Lower)?;
    let upper = eng.run_prefix(&seed, relu_op, LaneDir::Upper)?;
    stats.dense_pairs += 1;
    Ok((eng.concretize_lower(&lower), eng.concretize_upper(&upper)))
}

/// #joint-refresh: carry a layer's retained clip rows onto a refreshed `unst`.
///
/// `ClipRows` is indexed by POSITION in `unst`. A refresh only ever SHRINKS
/// boxes, so a stable neuron stays stable and the refreshed `unst` is a
/// SUBSEQUENCE of the old one; every surviving neuron keeps exactly its own
/// row. The rows themselves stay valid on a tightened box: their `sl_lo` /
/// `sl_up` slacks bound the LINE against the true pre-activation pointwise
/// (calibrated on the forward-only bounds at retention, `root.rs` `bi_cal`),
/// which no later box intersection changes.
///
/// `None` when the shapes disagree or the new list is not a subsequence: the
/// layer then carries no rows, and every clip consumer skips a layer without
/// rows — always sound.
pub(crate) fn filter_clip_rows(
    rows: &ClipRows,
    old_unst: &[usize],
    new_unst: &[usize],
) -> Option<ClipRows> {
    let n = old_unst.len();
    if rows.m.nrows() != n
        || rows.d.nrows() != n
        || rows.sl_lo.len() != n
        || rows.sl_up.len() != n
        || rows.bm.len() != n
        || rows.bd.len() != n
    {
        return None;
    }
    if new_unst == old_unst {
        return Some(rows.clone());
    }
    let mut keep = Vec::with_capacity(new_unst.len());
    let mut position = 0usize;
    for &neuron in new_unst {
        while position < n && old_unst[position] != neuron {
            position += 1;
        }
        if position == n {
            return None;
        }
        keep.push(position);
        position += 1;
    }
    let pick = |values: &[f64]| keep.iter().map(|&p| values[p]).collect::<Vec<f64>>();
    Some(ClipRows {
        m: rows.m.select(Axis(0), &keep),
        d: rows.d.select(Axis(0), &keep),
        sl_lo: pick(&rows.sl_lo),
        sl_up: pick(&rows.sl_up),
        bm: pick(&rows.bm),
        bd: pick(&rows.bd),
    })
}

/// #joint-refresh (cct plan P5 step 2): re-derive the ROOT's intermediate
/// boxes from its CURRENT gates — the committed (optimized) alpha included —
/// with the lane's own certified backward engine, and return the refreshed
/// frame as a NEW `RootGates` (the input root is never mutated).
///
/// This is `#backward-interm` run again, AFTER the alpha ascent, over the
/// whole finished root instead of interleaved with the forward walk: per trunk
/// relu layer `li >= 1`, in order, the `topk` widest unstable neurons are
/// re-bounded by a prefix pass over the refreshed layers `0..li` (so the
/// tightening compounds down the trunk exactly as in the original phase), each
/// fresh box is INTERSECTED with the current one, and the touched neurons'
/// gates are re-certified. It is the backward-only half of alpha-CROWN's joint
/// loop (abc recomputes every intermediate box from the current alphas each
/// iteration, `interm_bounds=None`); the driver
/// (`rebake::maybe_joint_refresh_root`) alternates it with a short ascent that
/// rebuilds the frozen head frame (y-box, head gates, m1/m2v) from the
/// refreshed gates.
///
/// SOUNDNESS.
/// * Every fresh box is the unchanged certified engine's outward enclosure of
///   the pre-activation over the root box, under valid prefix relaxations
///   (any `alpha in [0, 1]` is a valid lower line; every chord was certified by
///   `repair_upper_lines` on a box that contains the true range). The current
///   box is a valid enclosure too, so their intersection is one; it is
///   SHRINK-ONLY by construction.
/// * A CROSSED intersection (two sound enclosures of a non-empty root range
///   cannot cross) is a defect signal, never an emptiness certificate: the
///   WHOLE refresh is refused and the caller keeps the previous root.
/// * Touched neurons re-derive their upper chord `(s, c)` through
///   `gates_from_box` + `repair_upper_lines` on the tightened box (valid there,
///   and the box contains the true range); a still-unstable neuron keeps its
///   committed alpha (valid for any value in `[0, 1]`, clamped); a neuron that
///   became stable takes the exact line (`1, 1, 0` / `0, 0, 0`) and leaves
///   `unst`; `ms = max(alpha, s)` is re-derived for every touched neuron — a
///   stale `ms` under a moved `s` would under-scale the carried error.
/// * Stopping at the wall slice, skipping a non-finite column, or refusing a
///   layer the crop cannot express all leave the corresponding box as it was:
///   always sound, only looser.
/// * Clip rows follow their neurons (`filter_clip_rows`).
///
/// The candidate's published bound is NOT trusted from here: the driver scores
/// it with the unchanged certified `root_eval` and accepts it only on a strict
/// improvement.
///
/// `gpu` (#bi-gpu): the joint refresh's device route and accounting, shared
/// across its rounds; `None` (dark) runs every pair exactly as before.
pub(crate) fn refresh_root_intermediates(
    net: &TwinNet,
    root: &RootGates,
    topk: usize,
    chunk: usize,
    window: bool,
    stop_at: Option<Instant>,
    gpu: Option<&mut BiGpu>,
) -> Result<(RootGates, JointRefreshStats)> {
    refresh_root_intermediates_with_patch(net, root, topk, chunk, window, None, stop_at, gpu)
}

/// [`refresh_root_intermediates`] with its prefix pairs run as #bi-patch
/// per-column receptive-field walks when `patch` is `Some` (the frozen
/// phase policy's `BiPatchRuntimePolicy::cfg`). `None` is
/// [`refresh_root_intermediates`] exactly. Every patch pair is the dense pair
/// restricted to its support (never looser), so the refresh's shrink-only
/// contract is unchanged; a layer or chunk the plan cannot express runs the
/// crop (when `window`) or the dense pair.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refresh_root_intermediates_with_patch(
    net: &TwinNet,
    root: &RootGates,
    topk: usize,
    chunk: usize,
    window: bool,
    patch: Option<super::patch::PatchCfg>,
    stop_at: Option<Instant>,
    gpu: Option<&mut BiGpu>,
) -> Result<(RootGates, JointRefreshStats)> {
    if !root.mode.outward() {
        return Err(NyError::InvalidSpec(
            "margin_row: joint intermediate refresh requires outward mode".into(),
        ));
    }
    refresh_intermediates_impl(net, root, topk, chunk, window, patch, stop_at, gpu)
}

/// #root-certify-once: the SAME refresh over an UNTRUSTED Parity scratch root —
/// abc's per-iteration intermediate recomputation, run as a SEARCH step.
///
/// Every prefix pass runs in the scratch's own mode (`RoundMode::Parity`: no
/// error lane, no outward rounding), so the refreshed boxes are ESTIMATES that
/// steer the untrusted alpha ascent and nothing else. They are never an
/// enclosure anyone relies on: the scratch is never read by a verdict path, and
/// `root_certify_once` certifies the chosen alpha against the certified root's
/// own boxes (or against a CERTIFIED re-run of `refresh_root_intermediates`).
/// It REFUSES an Outward root, so it can never publish an uncertified box into
/// a certified frame. A crossed intersection still refuses the whole refresh
/// (the search then keeps its previous frame).
pub(crate) fn refresh_untrusted_intermediates(
    net: &TwinNet,
    scratch: &RootGates,
    topk: usize,
    chunk: usize,
    window: bool,
    stop_at: Option<Instant>,
) -> Result<(RootGates, JointRefreshStats)> {
    if scratch.mode.outward() {
        return Err(NyError::InvalidSpec(
            "margin_row: the untrusted intermediate refresh refuses a certified (outward) root"
                .into(),
        ));
    }
    // The untrusted search step never takes the #bi-gpu device route: its boxes
    // are estimates, so there is nothing to certify on the device. Nor the
    // #bi-patch walk (its certified refreshes are the ones patch mode serves).
    refresh_intermediates_impl(net, scratch, topk, chunk, window, None, stop_at, None)
}

/// The refresh body shared by the certified and the untrusted entry. Every
/// pass and every re-derived gate is in `root.mode`; the two entries decide
/// which mode is admissible, this body never does.
#[allow(clippy::too_many_arguments)]
fn refresh_intermediates_impl(
    net: &TwinNet,
    root: &RootGates,
    topk: usize,
    chunk: usize,
    window: bool,
    patch: Option<super::patch::PatchCfg>,
    stop_at: Option<Instant>,
    mut gpu: Option<&mut BiGpu>,
) -> Result<(RootGates, JointRefreshStats)> {
    if root.spec_alpha_policy.is_some() {
        return Err(NyError::InvalidSpec(
            "margin_row: joint intermediate refresh must run before #spec-alpha is installed"
                .into(),
        ));
    }
    let t0 = Instant::now();
    let topk = topk.max(1);
    let chunk = chunk.max(1);
    let mut stats = JointRefreshStats::default();
    // Compact certified frame. Clip rows are attached at the end, filtered to
    // the refreshed `unst`, so a refused refresh never copies them.
    let mut cand = RootGates {
        mode: root.mode,
        mid: root.mid.clone(),
        rad: root.rad.clone(),
        xabs: root.xabs.clone(),
        layers: root
            .layers
            .iter()
            .map(|layer| LayerGates {
                op: layer.op,
                n: layer.n,
                l: layer.l.clone(),
                u: layer.u.clone(),
                alpha: layer.alpha.clone(),
                s: layer.s.clone(),
                c: layer.c.clone(),
                ms: layer.ms.clone(),
                unst: layer.unst.clone(),
                clip_rows: None,
            })
            .collect(),
        lo: root.lo.clone(),
        hi: root.hi.clone(),
        backward_interm_execution: root.backward_interm_execution,
        root_precision_policy: root.root_precision_policy,
        root_precision_execution: root.root_precision_execution,
        // Refused above unless `None`: the refresh precedes the axis install.
        spec_alpha_policy: None,
        // Head slopes are valid on any box (consumed only where a head neuron
        // is unstable in the box being evaluated).
        head_alpha: root.head_alpha.clone(),
        // The refreshed gates differ; nothing is inherited.
        root_eval_memo: super::bab::RootEvalMemo::default(),
        // Dense refreshes use the repaired per-column arithmetic. Patch
        // refreshes retain a separate legacy error producer, so their gates
        // cannot authorize the optional joint head source, even on fallback.
        head_group_source: if patch.is_none() {
            root.head_group_source.clone()
        } else {
            None
        },
    };
    let expired = || stop_at.is_some_and(|at| Instant::now() >= at);
    // Layer 0's prefix is affine: its forward box is already the exact linear
    // enclosure and a backward pass can only reproduce it.
    for li in 1..cand.layers.len() {
        if expired() {
            stats.deadline_cut = true;
            break;
        }
        let (relu_op, n_pre, mut cols) = {
            let layer = &cand.layers[li];
            let mut sel: Vec<(f64, usize)> = layer
                .unst
                .iter()
                .filter(|&&j| j < layer.n)
                .map(|&j| (layer.u[j] - layer.l[j], j))
                .filter(|(width, _)| width.is_finite() && *width > 0.0)
                .collect();
            sel.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            sel.truncate(topk);
            let mut cols: Vec<usize> = sel.into_iter().map(|(_, j)| j).collect();
            cols.sort_unstable();
            (layer.op, layer.n, cols)
        };
        if cols.is_empty() {
            continue;
        }
        stats.layers += 1;
        // #bi-patch: the layer's per-column receptive-field plan (dark:
        // `None`, never built). A planned layer never enters the crop.
        let plan = match patch {
            Some(pcfg) => match super::patch::PatchPlan::build(net, relu_op, pcfg.ratio) {
                Ok(plan) => Some(plan),
                Err(_) => {
                    stats.patch_fallbacks += 1;
                    None
                }
            },
            None => None,
        };
        let patch_cap: Option<usize> = match (plan.as_ref(), patch) {
            (Some(plan), Some(pcfg)) => Some(plan.chunk_cap(pcfg.mb)),
            _ => None,
        };
        let shapes = if window && plan.is_none() {
            let shapes = window::prefix_shapes(net, relu_op);
            if shapes.is_none() {
                stats.window_fallbacks += 1;
            }
            shapes
        } else {
            None
        };
        let run_ends: Option<Vec<usize>> = match (shapes.as_deref(), net.ops.get(relu_op)) {
            (Some(sh), Some(TwinOp::Relu { input, .. })) if *input < sh.len() => {
                let (ordered, ends) = window::tile_order(&cols, sh[*input], WINDOW_TARGET_COLS);
                cols = ordered;
                Some(ends)
            }
            _ => None,
        };
        let mut fresh: Vec<(usize, f64, f64)> = Vec::with_capacity(cols.len());
        let mut cursor = 0usize;
        while cursor < cols.len() {
            if expired() {
                stats.deadline_cut = true;
                break;
            }
            let mut width = chunk.min(cols.len() - cursor);
            if let Some(ends) = run_ends.as_deref() {
                width = width.min(ends[cursor].saturating_sub(cursor)).max(1);
            }
            if let Some(cap) = patch_cap {
                width = width.min(cap).max(1);
            }
            let part = &cols[cursor..cursor + width];
            let (lb, ub) = refresh_prefix_pair(
                net,
                &cand,
                relu_op,
                n_pre,
                part,
                shapes.as_deref(),
                plan.as_ref(),
                &mut stats,
                gpu.as_deref_mut(),
            )?;
            for (column, &j) in part.iter().enumerate() {
                fresh.push((j, lb[column], ub[column]));
            }
            stats.columns += width;
            cursor += width;
        }
        // Apply this layer's fresh boxes (a partial layer, when the slice
        // expired mid-layer, is applied too: each box is certified alone).
        let layer = &mut cand.layers[li];
        let mut touched = Vec::new();
        for (j, lb, ub) in fresh {
            if !(lb.is_finite() && ub.is_finite()) {
                stats.nonfinite += 1;
                continue;
            }
            let (nl, nu) = (layer.l[j].max(lb), layer.u[j].min(ub));
            if nl > nu {
                return Err(NyError::NumericalInstability(format!(
                    "margin_row: joint refresh interval crossed at ({li},{j}): \
                     current=[{},{}] fresh=[{lb},{ub}] — refusing the whole refresh",
                    layer.l[j], layer.u[j]
                )));
            }
            if nl > layer.l[j] || nu < layer.u[j] {
                let was_unstable = layer.l[j] < 0.0 && layer.u[j] > 0.0;
                layer.l[j] = nl;
                layer.u[j] = nu;
                stats.tightened += 1;
                if was_unstable && (nl >= 0.0 || nu <= 0.0) {
                    stats.stabilized += 1;
                }
                touched.push(j);
            }
        }
        if !touched.is_empty() {
            let (alpha_h, mut s, mut c) = super::root::gates_from_box(&layer.l, &layer.u);
            super::root::repair_upper_lines(&layer.l, &layer.u, &mut s, &mut c);
            for &j in &touched {
                let unstable = layer.l[j] < 0.0 && layer.u[j] > 0.0;
                layer.s[j] = s[j];
                layer.c[j] = c[j];
                layer.alpha[j] = if unstable {
                    layer.alpha[j].clamp(0.0, 1.0)
                } else {
                    alpha_h[j]
                };
                layer.ms[j] = layer.alpha[j].max(layer.s[j]);
            }
            let (l, u) = (&layer.l, &layer.u);
            layer.unst.retain(|&j| l[j] < 0.0 && u[j] > 0.0);
        }
        if stats.deadline_cut {
            break;
        }
    }
    for (li, layer) in cand.layers.iter_mut().enumerate() {
        if let Some(rows) = root.layers[li].clip_rows.as_ref() {
            layer.clip_rows = filter_clip_rows(rows, &root.layers[li].unst, &layer.unst);
        }
    }
    stats.secs = t0.elapsed().as_secs_f64();
    Ok((cand, stats))
}

impl BiState {
    pub(crate) fn new(
        policy: super::BackwardIntermRuntimePolicy,
        deadline: Option<Instant>,
    ) -> Self {
        let t0 = Instant::now();
        let cfg = policy.config(RoundMode::Outward).unwrap_or(BiCfg {
            yield_window: 0,
            yield_ratio: 0.0,
            secs: policy.secs,
            chunk: policy.chunk,
            chunk_secs: policy.chunk_secs,
            topk: policy.topk,
            interm_alpha: policy.interm_alpha,
        });
        let budget = match deadline {
            Some(dl) => cfg
                .secs
                .min(0.4 * dl.saturating_duration_since(t0).as_secs_f64()),
            None => cfg.secs,
        };
        let deadline_capped = policy.enabled && budget < cfg.secs;
        let effective_chunk_secs = if cfg.chunk_secs > 0.0 {
            cfg.chunk_secs.min((0.1 * budget).max(1e-9))
        } else {
            0.0
        };
        let chunk_secs_capped = cfg.chunk_secs > 0.0 && effective_chunk_secs < cfg.chunk_secs;
        if policy.enabled {
            // Engagement telemetry (R9): armed must be visible in one log line
            // even when every layer is later skipped.
            eprintln!(
                "[backward-interm] armed: budget={budget:.1}s chunk={} chunk_secs={:.3} \
                 effective_chunk_secs={effective_chunk_secs:.3} chunk_secs_capped={} topk={}",
                cfg.chunk, cfg.chunk_secs, chunk_secs_capped, cfg.topk
            );
            // #interm-alpha R9: an armed dose announces itself BEFORE any layer
            // can decline, so an inert arming is one grep away. A null from a
            // lever whose telemetry never printed is VOID, not negative.
            if cfg.interm_alpha > 0 {
                eprintln!(
                    "[interm-alpha] armed: dose={} policies={} (per-start-node prefix alpha)",
                    cfg.interm_alpha,
                    cfg.interm_alpha.min(INTERM_ALPHA_POLICIES)
                );
            }
            // #bi-window R9: an armed crop announces itself before any layer
            // can decline, so an inert arming is one grep away.
            if policy.window {
                eprintln!(
                    "[bi-window] armed: target_cols={WINDOW_TARGET_COLS} (receptive-field-cropped \
                     prefix passes; dense fallback wherever the prefix is not a single-consumer \
                     conv/affine/relu chain)"
                );
            }
            // #bi-patch R9: likewise, before any layer can refuse its plan.
            // `rho=` is the crossover, spelled as on the per-layer plan line,
            // where `ratio=` is the dense/patch MAC ratio.
            if let Some(pcfg) = policy.patch.cfg() {
                eprintln!(
                    "[bi-patch] armed: rho={:.3} mb={} (per-column receptive-field patches; \
                     crop/dense fallback)",
                    pcfg.ratio, pcfg.mb
                );
            }
        }
        // #bi-gpu R9: the armed state prints `[bi-gpu] armed` here and its
        // `[bi-gpu] done` receipt when it drops. Dark (the default): `None`.
        // R2: `BiGpu::armed` REFUSES outside tests (`[bi-gpu] refused`), so an
        // env-armed flag leaves this `None` and the phase dark, bit for bit.
        let bi_gpu = if policy.enabled && policy.gpu {
            BiGpu::armed(BiGpuScope::BackwardInterm)
        } else {
            None
        };
        Self {
            yield_last_tightened: 0,
            yield_last_secs: 0.0,
            yield_last_chunks: 0,
            yield_layers_done: 0,
            yield_best_rate: 0.0,
            yield_stopped: false,
            // #router-r3 `rule.ia_yield`: the acting optimizer's stop
            // authority arms the candidate loop's governor (0.15 after 2
            // layers); unarmed it is the shipped constant (`None`), so the
            // phase is byte-identical.
            ia_yield: {
                let (ratio, min_samples, _) = super::marginal::ia_governor();
                super::marginal::MarginalYield::new(ratio, min_samples)
            },
            ia_yield_stopped: false,
            ia_yield_last_secs: 0.0,
            ia_yield_fired_layer: None,
            ia_yield_fired_rate: 0.0,
            enabled: policy.enabled,
            cfg,
            t0,
            budget,
            effective_chunk_secs,
            chunk_secs_capped,
            deadline,
            exhausted: false,
            tightened_total: 0,
            stabilized_total: 0,
            crossed_total: 0,
            pass_errors: 0,
            work_secs: 0.0,
            last_chunk_secs: 0.0,
            chunks_declined: 0,
            eligible_layers: 0,
            suppressed_setup_layers: 0,
            layers_started: 0,
            chunks_completed: 0,
            columns_completed: 0,
            deadline_capped,
            chunk_counterfactual_chunks: 0,
            chunk_ceiling_binding_chunks: 0,
            fixed_width_chunks: 0,
            adaptive_width_changes: 0,
            topk_binding_layers: 0,
            ia_tightened_total: 0,
            ia_stabilized_total: 0,
            ia_pass_pairs: 0,
            ia_pass_errors: 0,
            ia_secs: 0.0,
            ia_alpha_moved: [0; INTERM_ALPHA_POLICIES],
            err_lane: super::ErrLane::PerCol,
            window: policy.window,
            win_pairs: 0,
            win_fallbacks: 0,
            win_rows_window: 0,
            win_rows_dense: 0,
            bi_gpu,
            // #bi-patch: armed only on an enabled phase (the space.rs edge).
            patch: if policy.enabled {
                policy.patch.cfg()
            } else {
                None
            },
            patch_pairs: 0,
            patch_fused_cols: 0,
            patch_fallbacks: 0,
            patch_converted: 0,
            patch_macs_patch: 0.0,
            patch_macs_dense: 0.0,
            patch_rows_patch: 0,
            patch_rows_dense: 0,
            patch_secs: 0.0,
        }
    }

    /// #mr-errlane: carry the building root's certified error lane into this
    /// phase's scratch roots. Default `PerCol` keeps every existing caller
    /// (and every test constructing a `BiState` directly) byte-identical.
    #[must_use]
    pub(crate) fn with_err_lane(mut self, lane: super::ErrLane) -> Self {
        self.err_lane = lane;
        self
    }

    /// #bi-window: account one completed cropped pass pair.
    fn note_window_pair(&mut self, pair: &window::WindowedPair) {
        self.win_pairs += 1;
        self.win_rows_window = self.win_rows_window.saturating_add(pair.rows_window);
        self.win_rows_dense = self.win_rows_dense.saturating_add(pair.rows_dense);
    }

    /// #bi-patch: account one completed patch pair and its wall.
    fn note_patch_pair(&mut self, pair: &super::patch::PatchPair, secs: f64) {
        self.patch_pairs += 1;
        self.patch_fused_cols += pair.fused_cols;
        self.patch_converted += usize::from(pair.converted);
        self.patch_macs_patch += pair.macs_patch;
        self.patch_macs_dense += pair.macs_dense;
        self.patch_rows_patch = self.patch_rows_patch.saturating_add(pair.rows_patch);
        self.patch_rows_dense = self.patch_rows_dense.saturating_add(pair.rows_dense);
        self.patch_secs += secs;
    }

    /// #bi-gpu: RSS-admit ONE prefix pass pair to this phase's device route
    /// and open its accounting. `None` means the pair runs on the dark engine
    /// when #bi-gpu is dark (exactly the historical pair) and wholly on the
    /// CPU kernel when it is armed but the admission refused.
    fn gpu_pair_open(&mut self) -> Option<PairMark> {
        self.bi_gpu.as_mut()?.open_pair()
    }

    /// #bi-gpu: close an admitted pair's accounting (no-op when dark).
    fn gpu_pair_close(&mut self, mark: Option<PairMark>) {
        if let (Some(gpu), Some(mark)) = (self.bi_gpu.as_mut(), mark) {
            gpu.close_pair(mark);
        }
    }

    /// #bi-gpu test injection: replace the phase's device route (e.g. with an
    /// emulated engine), or remove it.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_bi_gpu(mut self, gpu: Option<BiGpu>) -> Self {
        self.bi_gpu = gpu;
        self
    }

    /// #bi-gpu test surface: the phase's device-route totals (`None` dark).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn bi_gpu_stats(&self) -> Option<bi_gpu::BiGpuStats> {
        self.bi_gpu.as_ref().map(BiGpu::stats)
    }

    /// #bi-window test surface: cropped pairs completed and dense fallbacks.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn window_counts(&self) -> (usize, usize) {
        (self.win_pairs, self.win_fallbacks)
    }

    /// #bi-patch test surface: patch pairs completed and fallbacks.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn patch_counts(&self) -> (usize, usize) {
        (self.patch_pairs, self.patch_fallbacks)
    }

    /// #bi-patch test surface: whether the frozen policy armed patch mode.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn patch_armed(&self) -> bool {
        self.patch.is_some()
    }

    fn out_of_time(&mut self) -> bool {
        if self.exhausted {
            return true;
        }
        if self.t0.elapsed().as_secs_f64() >= self.budget {
            self.exhausted = true;
            return true;
        }
        if let Some(dl) = self.deadline {
            // Leave headroom for the rest of the forward build itself.
            if Instant::now() + std::time::Duration::from_millis(250) > dl {
                self.exhausted = true;
                return true;
            }
        }
        // #bi-yield: last, so every pre-existing termination reason keeps its
        // exact precedence and the OFF path never reaches this call.
        if self.yield_collapsed() {
            self.exhausted = true;
            return true;
        }
        false
    }

    /// #bi-yield: has the marginal return collapsed?
    ///
    /// # Why this exists
    ///
    /// The phase had a wall budget and NO notion of whether it was still
    /// earning anything, so it spent its cap unconditionally. Measured
    /// 2026-09-16: on cifar100 resnet_large the root build costs 35.9-41.0 s of
    /// a 100 s budget and exhausts this phase's 40 s cap on 3 of 4 rows, after
    /// which BaB explores 0-40 domains. alpha-beta-CROWN runs NO root
    /// intermediate-refinement phase at all on this benchmark
    /// (`references/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp25/cifar100.yaml`
    /// — alpha-crown `iteration: 20` then straight to BaB) and proves the same
    /// rows in 15-19 s. The same constant costs only 8.0-8.2 s on
    /// resnet_medium, so it is a CATEGORY-scoped constant tuned on one of the
    /// two models the category contains.
    ///
    /// # Soundness
    ///
    /// Stopping early can only publish a LOOSER intermediate box. The phase is a
    /// shrink-only intersection (module docs above), so ending it sooner leaves
    /// bounds at their pre-tightening values — never wrong, only wider. It
    /// cannot turn an unsat into a sat; it trades root tightness for tree time.
    fn yield_collapsed(&mut self) -> bool {
        if self.cfg.yield_window == 0 {
            return false; // OFF: byte-identical to the fixed-budget phase.
        }
        let layers = self.yield_layers_done;
        if layers < self.yield_last_chunks + self.cfg.yield_window {
            return false; // not a full window yet
        }
        let d_tight = self
            .tightened_total
            .saturating_sub(self.yield_last_tightened);
        let d_secs = (self.work_secs - self.yield_last_secs).max(1e-9);
        let rate = d_tight as f64 / d_secs;
        self.yield_last_tightened = self.tightened_total;
        self.yield_last_secs = self.work_secs;
        self.yield_last_chunks = layers;
        if rate > self.yield_best_rate {
            self.yield_best_rate = rate;
            return false;
        }
        // The first window sets the baseline; it can never trigger a stop.
        if self.yield_best_rate <= 0.0 {
            return false;
        }
        if rate < self.cfg.yield_ratio * self.yield_best_rate {
            let returned = (self.budget - self.t0.elapsed().as_secs_f64()).max(0.0);
            eprintln!(
                "[bi-yield] STOP after {layers} layers: window rate={rate:.1} tightened/s vs \
                 best={:.1} (ratio {:.3} < {:.3}); returning {returned:.1}s of the {:.1}s \
                 budget to the tree search",
                self.yield_best_rate,
                rate / self.yield_best_rate,
                self.cfg.yield_ratio,
                self.budget,
            );
            self.yield_stopped = true;
            return true;
        }
        false
    }

    /// Whether an enabled counterfactual would enter governed selection/setup
    /// at this point. This is read-only so an OFF receipt can prove causal wall
    /// suppression without mutating clock state or executing bound work.
    fn would_enter_governed_setup_now(&self) -> bool {
        let now = Instant::now();
        self.t0.elapsed().as_secs_f64() < self.budget
            && self
                .deadline
                .is_none_or(|deadline| now + std::time::Duration::from_millis(250) <= deadline)
    }

    /// Conservative proof that TOPK, rather than the clock, stopped at least
    /// one omitted column. This is evidence-only and never mutates admission
    /// state: a missing estimate or marginal headroom earns no T1 credit.
    fn can_admit_omitted_columns(
        &self,
        seconds_per_col: Option<f64>,
        omitted_columns: usize,
    ) -> bool {
        if self.exhausted || self.last_chunk_secs <= 0.0 {
            return false;
        }
        let now = Instant::now();
        let spent = self.t0.elapsed().as_secs_f64();
        if spent >= self.budget
            || self
                .deadline
                .is_some_and(|deadline| now + std::time::Duration::from_millis(250) > deadline)
        {
            return false;
        }
        let width = next_chunk_width(
            self.effective_chunk_secs,
            seconds_per_col,
            self.cfg.chunk,
            omitted_columns,
        );
        predicted_chunk_secs(self.last_chunk_secs, seconds_per_col, width)
            .is_some_and(|predicted| self.budget - spent >= predicted)
    }

    /// Tighten one trunk ReLU's forward box `(l, u)` in place via prefix
    /// backward passes. `layers` are the gates frozen SO FAR (exactly the
    /// prefix); they are moved into a temporary [`RootGates`] for the engine
    /// and always restored. Fail-open everywhere: on any refusal the forward
    /// bounds stand untouched.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tighten_layer(
        &mut self,
        net: &TwinNet,
        mode: RoundMode,
        mid: &[f64],
        rad: &[f64],
        xabs: &[f64],
        lo: &[f64],
        hi: &[f64],
        layers: &mut Vec<LayerGates>,
        relu_op: usize,
        l: &mut [f64],
        u: &mut [f64],
    ) {
        let li = layers.len();
        if li == 0 {
            // Purely affine prefix: the forward tableau is already the exact
            // linear enclosure here; a backward pass can only reproduce it.
            return;
        }
        // A disabled policy still observes eligibility so a scored kill-switch
        // arm can prove that it suppressed live work rather than missing the
        // mechanism structurally. It never sorts, allocates, or moves a bound.
        if !self.enabled {
            if l.iter()
                .zip(u.iter())
                .any(|(&lower, &upper)| lower < 0.0 && upper > 0.0)
            {
                self.eligible_layers += 1;
                self.suppressed_setup_layers += usize::from(self.would_enter_governed_setup_now());
            }
            return;
        }
        // Widest unstable neurons first: they carry the loosest relaxation
        // triangles, which is what the tightened gates buy back downstream.
        let mut sel: Vec<(f64, usize)> = (0..l.len())
            .filter(|&j| l[j] < 0.0 && u[j] > 0.0)
            .map(|j| (u[j] - l[j], j))
            .collect();
        if sel.is_empty() {
            return;
        }
        self.eligible_layers += 1;
        if self.out_of_time() {
            return;
        }
        let unst_before = sel.len();
        sel.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        sel.truncate(self.cfg.topk);
        let topk_would_bind = unst_before > sel.len();
        // Ascending neuron order for cache-friendly seed construction.
        let mut cols: Vec<usize> = sel.into_iter().map(|(_, j)| j).collect();
        cols.sort_unstable();
        self.layers_started += 1;

        // #bi-patch: the per-column receptive-field plan, built ONCE per layer
        // and shared by every chunk and every #interm-alpha candidate. Dark
        // (the default) never enters this block: `patch_plan` stays `None` and
        // every pass below is exactly the pre-#bi-patch pass. Armed on a layer
        // the planner cannot express (a transposed conv, an unequal-stride
        // union, no gain), the refusal is printed once and the layer runs as
        // the phase runs without patch mode (the crop when armed, else dense).
        let patch_plan: Option<super::patch::PatchPlan> = match self.patch {
            Some(pcfg) => match super::patch::PatchPlan::build(net, relu_op, pcfg.ratio) {
                Ok(plan) => {
                    eprintln!("{}", plan.plan_line(li, cols.len(), pcfg.ratio));
                    Some(plan)
                }
                Err(why) => {
                    self.patch_fallbacks += 1;
                    eprintln!(
                        "{}",
                        super::patch::PatchPlan::refused_line(
                            li,
                            relu_op,
                            cols.len(),
                            pcfg.ratio,
                            why
                        )
                    );
                    None
                }
            },
            None => None,
        };
        // #bi-patch: the widest chunk whose fused patch walk fits the
        // working-set cap. Dark: `None`, and every width is the historical one.
        let patch_cap: Option<usize> = match (patch_plan.as_ref(), self.patch) {
            (Some(plan), Some(pcfg)) => Some(plan.chunk_cap(pcfg.mb)),
            _ => None,
        };

        // #bi-window: the prefix geometry, and a TILE-MAJOR column order so
        // each chunk's receptive field is one compact rectangle. Dark (the
        // default) never enters this block: `win_shapes`/`win_runs` stay
        // `None`, the historical ascending order stands, and every pass below
        // is the dense pass, byte-identical. Armed on a layer the crop can
        // never express (a residual prefix), `prefix_shapes` refuses ONCE here
        // and the layer runs exactly as dark — not as tile-capped dense chunks
        // behind a crop every chunk would refuse. A layer #bi-patch planned
        // never enters the crop (patch mode ignores tile order).
        let win_shapes: Option<Vec<window::Shape>> = if self.window && patch_plan.is_none() {
            let shapes = window::prefix_shapes(net, relu_op);
            if shapes.is_none() {
                self.win_fallbacks += 1;
            }
            shapes
        } else {
            None
        };
        let win_runs: Option<Vec<usize>> = match (win_shapes.as_deref(), net.ops.get(relu_op)) {
            (Some(shapes), Some(TwinOp::Relu { input, .. })) if *input < shapes.len() => {
                let (ordered, run_ends) =
                    window::tile_order(&cols, shapes[*input], WINDOW_TARGET_COLS);
                cols = ordered;
                Some(run_ends)
            }
            _ => None,
        };

        let layer_t0 = Instant::now();
        // Temporary root over the frozen prefix. The engine only reads
        // `layers[0..li]` (every relu in the prefix has a smaller layer
        // index), `mid`/`rad`/`xabs` for concretization, and `mode`.
        let tmp = RootGates {
            mode,
            mid: mid.to_vec(),
            rad: rad.to_vec(),
            xabs: xabs.to_vec(),
            layers: std::mem::take(layers),
            lo: lo.to_vec(),
            hi: hi.to_vec(),
            backward_interm_execution: BackwardIntermExecutionEvidence::default(),
            // #mr-errlane: only the error lane is inherited; the scratch root
            // never builds a tableau, so the precision flags stay default.
            root_precision_policy: super::RootPrecisionRuntimePolicy {
                err_lane: self.err_lane,
                ..super::RootPrecisionRuntimePolicy::default()
            },
            root_precision_execution: super::root::RootPrecisionExecutionEvidence::default(),
            // Prefix-only scratch root: the axis lives on the root direct pass.
            spec_alpha_policy: None,
            // Prefix-only scratch root: never reaches the head.
            head_alpha: None,
            // Prefix-only scratch root: never evaluated by `root_eval`.
            root_eval_memo: super::bab::RootEvalMemo::default(),
            head_group_source: None,
        };
        let (mut tightened, mut stabilized, mut crossed, mut done) =
            (0usize, 0usize, 0usize, 0usize);
        // #interm-alpha: this layer's marginal contribution from the
        // per-start-node candidates alone.
        let (mut ia_tightened, mut ia_stabilized) = (0usize, 0usize);
        let mut seconds_per_col: Option<f64> = None;
        {
            let eng = BackwardEngine::new(net, &tmp);
            // #bi-gpu: the same engine over this phase's scoped device route,
            // and a CPU-only one for the pairs the RSS admission refuses, both
            // built only when armed. Dark, `gpu_route` / `eng_gpu` / `eng_cpu`
            // are `None`, no pair is ever admitted, and every pass below runs
            // on `eng` exactly as before (byte-identical).
            let gpu_route: Option<GpuConvRoute> = self.bi_gpu.as_ref().map(BiGpu::route);
            let eng_gpu: Option<BackwardEngine<'_>> = gpu_route
                .as_ref()
                .map(|route| bi_gpu::engine(net, &tmp, Some(route)));
            let cpu_route = GpuConvRoute::Off;
            let eng_cpu: Option<BackwardEngine<'_>> = gpu_route
                .as_ref()
                .map(|_| bi_gpu::engine(net, &tmp, Some(&cpu_route)));
            let n_pre = l.len();
            let target_secs = self.effective_chunk_secs;
            let mut cursor = 0usize;
            'chunks: while cursor < cols.len() {
                if self.out_of_time() {
                    break;
                }
                // #bi-adaptive-chunk. The first one-column pass measures this
                // layer's actual prefix rate; later widths target the governed
                // wall slice and never exceed the fixed chunk's memory ceiling.
                // With target_secs=0 this selects the historical slices exactly.
                let width = next_chunk_width(
                    target_secs,
                    seconds_per_col,
                    self.cfg.chunk,
                    cols.len() - cursor,
                );
                // #bi-window: never let a chunk straddle two tiles (its crop
                // would be the union rectangle). Dark: `win_runs` is `None`
                // and the width is exactly the historical one.
                let width = match win_runs.as_deref() {
                    Some(run_ends) => width.min(run_ends[cursor].saturating_sub(cursor)).max(1),
                    None => width,
                };
                // #bi-patch: never exceed the plan's working-set cap. Dark:
                // `patch_cap` is `None` and the width is the historical one.
                let width = match patch_cap {
                    Some(cap) => width.min(cap).max(1),
                    None => width,
                };
                let remaining_cols = cols.len() - cursor;
                let chunk_is_counterfactual = chunk_width_differs_from_default(
                    target_secs,
                    seconds_per_col,
                    self.cfg.chunk,
                    remaining_cols,
                );
                let chunk_ceiling_is_binding = chunk_ceiling_binds_width(
                    target_secs,
                    seconds_per_col,
                    self.cfg.chunk,
                    remaining_cols,
                );
                let fixed_width_chunk = target_secs == 0.0 && width > 1;
                let adaptive_width_changed = target_secs > 0.0
                    && seconds_per_col.is_some()
                    && width != self.cfg.chunk.min(remaining_cols);
                // #bi-chunk-predict. Refuse to START a chunk the remaining
                // budget cannot pay for. Adaptive admission prices the width
                // ABOUT TO RUN, including a probe-to-wide jump; fixed-grain
                // admission retains the historical last-completed estimate.
                // This bounds overshoot to a layer's first probe (unpredictable
                // by construction) instead of every full-width chunk.
                //
                // SOUND BY THE MODULE'S OWN CONTRACT (see the header): skipping
                // any layer, chunk or neuron is always sound, because the
                // forward box stands and this phase only ever INTERSECTS
                // shrink-only. Declining costs tightness, never a proof.
                //
                // Deliberately NOT scaled by a safety factor: chunk costs
                // within a layer are near-uniform (identical seed shape, same
                // prefix), so `last` is a good estimator, and inflating it
                // would decline chunks that would in fact have fit.
                if let Some(predicted_secs) =
                    predicted_chunk_secs(self.last_chunk_secs, seconds_per_col, width)
                {
                    let spent = self.t0.elapsed().as_secs_f64();
                    let remaining = self.budget - spent;
                    if remaining < predicted_secs {
                        self.chunks_declined += 1;
                        self.exhausted = true;
                        break 'chunks;
                    }
                }
                let end = cursor + width;
                let chunk = &cols[cursor..end];
                let chunk_t0 = Instant::now();
                let w = chunk.len();
                // #bi-gpu: RSS-admit this chunk's baseline pair (cropped or
                // dense) to the device route. Dark: the dark engine; refused:
                // the CPU-only engine (`bi_gpu::pick`).
                let gpu_mark = self.gpu_pair_open();
                let (pair_route, pair_eng) = bi_gpu::pick(
                    gpu_mark.is_some(),
                    gpu_route.as_ref(),
                    eng_gpu.as_ref(),
                    &cpu_route,
                    eng_cpu.as_ref(),
                    &eng,
                );
                // #bi-patch: the per-column receptive-field pair first. `None`
                // means the plan cannot express this chunk and the crop (when
                // armed) or the dense pair runs instead; dark, `patch_plan` is
                // `None` and this is never attempted.
                let patch_t0 = patch_plan.as_ref().map(|_| Instant::now());
                let patched: Option<Result<(Vec<f64>, Vec<f64>)>> = match patch_plan
                    .as_ref()
                    .and_then(|plan| plan.prefix_bounds(net, &tmp, chunk, None, pair_route))
                {
                    Some(Ok(pair)) => {
                        let secs = patch_t0.map_or(0.0, |t| t.elapsed().as_secs_f64());
                        self.note_patch_pair(&pair, secs);
                        Some(Ok((pair.lb, pair.ub)))
                    }
                    Some(Err(error)) => Some(Err(error)),
                    None => {
                        if patch_plan.is_some() {
                            self.patch_fallbacks += 1;
                        }
                        None
                    }
                };
                // #bi-window: the receptive-field-cropped pair next. `None`
                // means the crop cannot express this chunk and the dense pair
                // below runs instead; dark, `win_shapes` is `None` and this is
                // never attempted.
                let windowed = if patched.is_some() {
                    None
                } else {
                    win_shapes.as_deref().and_then(|shapes| {
                        window::prefix_bounds(net, &tmp, relu_op, shapes, chunk, None, pair_route)
                    })
                };
                // The dense identity seed, built only when a dense pass runs
                // (always, when dark) and reused by the #interm-alpha
                // candidates below exactly as before.
                let mut dense_seed: Option<Seed> = None;
                let (lb, ub) = match (patched, windowed) {
                    (Some(Ok(bounds)), _) => bounds,
                    (Some(Err(_)), _) | (None, Some(Err(_))) => {
                        // Fail-open, exactly as the dense pair's refusal.
                        self.work_secs += chunk_t0.elapsed().as_secs_f64();
                        self.pass_errors += 1;
                        self.gpu_pair_close(gpu_mark);
                        break 'chunks;
                    }
                    (None, Some(Ok(pair))) => {
                        self.note_window_pair(&pair);
                        (pair.lb, pair.ub)
                    }
                    (None, None) => {
                        if win_shapes.is_some() {
                            self.win_fallbacks += 1;
                        }
                        let mut s = Array2::<f64>::zeros((n_pre, w));
                        for (c, &j) in chunk.iter().enumerate() {
                            s[[j, c]] = 1.0;
                        }
                        // Identity seed is exact: the engine zero-fills its error lane.
                        let seed = Seed { s, e: None };
                        let passes = (
                            pair_eng.run_prefix(&seed, relu_op, LaneDir::Lower),
                            pair_eng.run_prefix(&seed, relu_op, LaneDir::Upper),
                        );
                        let bounds = match passes {
                            (Ok(plo), Ok(pup)) => (
                                pair_eng.concretize_lower(&plo),
                                pair_eng.concretize_upper(&pup),
                            ),
                            _ => {
                                // Fail-open: an engine refusal here costs tightness
                                // only. Stop the layer (a failure would repeat).
                                self.work_secs += chunk_t0.elapsed().as_secs_f64();
                                self.pass_errors += 1;
                                self.gpu_pair_close(gpu_mark);
                                break 'chunks;
                            }
                        };
                        dense_seed = Some(seed);
                        bounds
                    }
                };
                self.gpu_pair_close(gpu_mark);
                // Include concretization in the adaptive rate: it is part of
                // the certified chunk's wall cost and scales with its width.
                let chunk_secs = chunk_t0.elapsed().as_secs_f64();
                self.work_secs += chunk_secs;
                for (c, &j) in chunk.iter().enumerate() {
                    let (lb_j, ub_j) = (lb[c], ub[c]);
                    if !(lb_j.is_finite() && ub_j.is_finite()) {
                        continue;
                    }
                    // Shrink-only intersection of two certified enclosures.
                    let nl = l[j].max(lb_j);
                    let nu = u[j].min(ub_j);
                    if nl > nu {
                        // Two sound enclosures of a non-empty range cannot
                        // cross: defect signal, NOT an emptiness certificate.
                        // Keep the forward bounds (fail-open) and shout.
                        crossed += 1;
                        continue;
                    }
                    if nl > l[j] || nu < u[j] {
                        tightened += 1;
                        if nl >= 0.0 || nu <= 0.0 {
                            stabilized += 1;
                        }
                        l[j] = nl;
                        u[j] = nu;
                    }
                }
                // ---- #interm-alpha: PER-START-NODE alpha candidates ----
                //
                // Re-bound this SAME chunk under prefix relaxations chosen for
                // THIS start node, and intersect again. Every candidate box is
                // produced by the unchanged certified engine under a valid
                // relaxation, so this is the module's established shrink-only
                // intersection with one more valid enclosure in it: the
                // published box is a SUBSET of what the pre-#interm-alpha
                // phase published, never larger. Dose 0 skips this block
                // entirely (byte-identical). The live risk is WALL, not
                // soundness — hence the out_of_time poll before each candidate
                // and the separate ia_secs charge.
                for policy in 0..self.cfg.interm_alpha.min(INTERM_ALPHA_POLICIES) {
                    // #ia-yield: the candidates stopped earning on an earlier
                    // layer. The baseline shrink-only pass above continues; only
                    // these extra enclosures are skipped, so the published box is
                    // the one the pre-#interm-alpha phase would have published on
                    // this layer -- looser, never wrong.
                    if self.ia_yield_stopped || self.out_of_time() {
                        break;
                    }
                    let Some((dg, alpha_moved)) = start_node_gates(&tmp, li, policy) else {
                        continue;
                    };
                    // Per-DOSE working-set size: `armed=1` proves the binary
                    // carries the lever; this proves the CONFIG gave it alphas.
                    self.ia_alpha_moved[policy] += alpha_moved;
                    let ia_t0 = Instant::now();
                    // #bi-gpu: RSS-admit the candidate's pair to the device
                    // route (dark: the dark engine; refused: the CPU-only one).
                    let ia_mark = self.gpu_pair_open();
                    let (ia_route, ia_eng) = bi_gpu::pick(
                        ia_mark.is_some(),
                        gpu_route.as_ref(),
                        eng_gpu.as_ref(),
                        &cpu_route,
                        eng_cpu.as_ref(),
                        &eng,
                    );
                    // #bi-patch: the candidate's pair runs as a patch walk
                    // too (the override's gates are read at each row's
                    // absolute neuron); `None` falls back below. Dark,
                    // `patch_plan` is `None` and this is never attempted.
                    let ia_patch_t0 = patch_plan.as_ref().map(|_| Instant::now());
                    let ia_patched = patch_plan
                        .as_ref()
                        .and_then(|plan| plan.prefix_bounds(net, &tmp, chunk, Some(&dg), ia_route));
                    let ia: Result<(Vec<f64>, Vec<f64>)> = match ia_patched {
                        Some(Ok(pair)) => {
                            let secs = ia_patch_t0.map_or(0.0, |t| t.elapsed().as_secs_f64());
                            self.note_patch_pair(&pair, secs);
                            Ok((pair.lb, pair.ub))
                        }
                        Some(Err(error)) => Err(error),
                        None => {
                            if patch_plan.is_some() {
                                self.patch_fallbacks += 1;
                            }
                            // #bi-window: the candidate's pair is cropped too
                            // (the override is cropped with the gates); dark,
                            // this is the historical dense pair on the same
                            // identity seed.
                            let ia_windowed = win_shapes.as_deref().and_then(|shapes| {
                                window::prefix_bounds(
                                    net,
                                    &tmp,
                                    relu_op,
                                    shapes,
                                    chunk,
                                    Some(&dg),
                                    ia_route,
                                )
                            });
                            match ia_windowed {
                                Some(Ok(pair)) => {
                                    self.note_window_pair(&pair);
                                    Ok((pair.lb, pair.ub))
                                }
                                Some(Err(error)) => Err(error),
                                None => {
                                    if win_shapes.is_some() {
                                        self.win_fallbacks += 1;
                                    }
                                    let seed = dense_seed.get_or_insert_with(|| {
                                        let mut s = Array2::<f64>::zeros((n_pre, w));
                                        for (c, &j) in chunk.iter().enumerate() {
                                            s[[j, c]] = 1.0;
                                        }
                                        Seed { s, e: None }
                                    });
                                    match (
                                        ia_eng.run_prefix_dom(
                                            seed,
                                            relu_op,
                                            LaneDir::Lower,
                                            Some(&dg),
                                        ),
                                        ia_eng.run_prefix_dom(
                                            seed,
                                            relu_op,
                                            LaneDir::Upper,
                                            Some(&dg),
                                        ),
                                    ) {
                                        (Ok(plo), Ok(pup)) => Ok((
                                            ia_eng.concretize_lower(&plo),
                                            ia_eng.concretize_upper(&pup),
                                        )),
                                        (Err(error), _) | (_, Err(error)) => Err(error),
                                    }
                                }
                            }
                        }
                    };
                    self.gpu_pair_close(ia_mark);
                    let (lb2, ub2) = match ia {
                        Ok(bounds) => bounds,
                        Err(_) => {
                            // Fail-open, exactly as the baseline pair: a
                            // refusal costs tightness only. Charge the wall and
                            // stop trying candidates on this chunk.
                            let spent = ia_t0.elapsed().as_secs_f64();
                            self.work_secs += spent;
                            self.ia_secs += spent;
                            self.ia_pass_errors += 1;
                            break;
                        }
                    };
                    let spent = ia_t0.elapsed().as_secs_f64();
                    self.work_secs += spent;
                    self.ia_secs += spent;
                    self.ia_pass_pairs += 1;
                    for (c, &j) in chunk.iter().enumerate() {
                        let (lb_j, ub_j) = (lb2[c], ub2[c]);
                        if !(lb_j.is_finite() && ub_j.is_finite()) {
                            continue;
                        }
                        // Count only what THIS candidate contributes beyond the
                        // baseline pass — the axis's marginal effect, not a
                        // restatement of the phase's.
                        let stable_before = l[j] >= 0.0 || u[j] <= 0.0;
                        let nl = l[j].max(lb_j);
                        let nu = u[j].min(ub_j);
                        if nl > nu {
                            crossed += 1;
                            continue;
                        }
                        if nl > l[j] || nu < u[j] {
                            ia_tightened += 1;
                            if !stable_before && (nl >= 0.0 || nu <= 0.0) {
                                ia_stabilized += 1;
                            }
                            l[j] = nl;
                            u[j] = nu;
                        }
                    }
                }
                done += w;
                self.chunks_completed += 1;
                self.columns_completed += w;
                self.chunk_counterfactual_chunks += usize::from(chunk_is_counterfactual);
                self.chunk_ceiling_binding_chunks += usize::from(chunk_ceiling_is_binding);
                self.fixed_width_chunks += usize::from(fixed_width_chunk);
                self.adaptive_width_changes += usize::from(adaptive_width_changed);
                // #bi-chunk-predict: the estimate for the NEXT chunk. Recorded
                // only on a chunk that ran to completion, so a fail-open break
                // never poisons it.
                self.last_chunk_secs = chunk_secs;
                seconds_per_col = Some((chunk_secs / w as f64).max(1e-9));
                cursor = end;
            }
        }
        *layers = tmp.layers;
        // Credit TOPK only when its selected candidate set was consumed in
        // full.  Merely truncating a vector before the budget cut is not
        // evidence that the cap changed any executed work.
        if topk_would_bind
            && done == cols.len()
            && self.can_admit_omitted_columns(seconds_per_col, unst_before - cols.len())
        {
            self.topk_binding_layers += 1;
        }
        if crossed > 0 {
            eprintln!(
                "[backward-interm] layer={li} CROSSED x{crossed}: forward/backward enclosures \
disagree — kept forward bounds (investigate; this should be impossible)"
            );
        }
        self.tightened_total += tightened;
        self.stabilized_total += stabilized;
        // #bi-yield: a LAYER is this phase's unit of work, so the marginal-rate
        // check belongs here rather than in the chunk-keyed `out_of_time` poll.
        // Setting `exhausted` makes every later `out_of_time` return true, so
        // the phase stops exactly as a budget exhaustion would.
        self.yield_layers_done += 1;
        if self.yield_collapsed() {
            self.exhausted = true;
        }
        self.crossed_total += crossed;
        self.ia_tightened_total += ia_tightened;
        self.ia_stabilized_total += ia_stabilized;
        // #ia-yield: one observation per layer, in tightenings PER SECOND of the
        // candidates' own charge. Placed before the layer's telemetry so the
        // line below can carry the resulting state.
        // A layer that charged NO candidate wall (out of time before the first
        // candidate, or no admitted chunk) carries no evidence about marginal
        // value. Observing it would divide 0 by the 1e-9 floor, read as a rate
        // of exactly 0, and end the lever on a layer it never ran -- an absence
        // treated as evidence. Skip it instead: fail closed, keep the candidates.
        let d_secs = self.ia_secs - self.ia_yield_last_secs;
        if self.cfg.interm_alpha > 0 && !self.ia_yield_stopped && d_secs > IA_YIELD_MIN_CHARGE_SECS
        {
            self.ia_yield_last_secs = self.ia_secs;
            let rate = ia_tightened as f64 / d_secs;
            if self.ia_yield.observe(rate) {
                self.ia_yield_stopped = true;
                self.ia_yield_fired_layer = Some(li);
                self.ia_yield_fired_rate = rate;
                eprintln!(
                    "[ia-yield] STOP after layer {li}: rate={rate:.1} < floor={:.1} \
                     extra_tightened/s (best={:.1}, ratio={:.3}, samples={}); the baseline \
                     shrink-only pass CONTINUES, the per-start-node candidates do not",
                    self.ia_yield.floor(),
                    self.ia_yield.best(),
                    self.ia_yield.ratio().unwrap_or(0.0),
                    self.ia_yield.samples(),
                );
            }
        }
        if self.cfg.interm_alpha > 0 {
            // The lever's OWN observable, per layer: what the per-start-node
            // alpha bought beyond the shared heuristic on this start node.
            // `extra_*=0` everywhere with a NON-flat alpha_moved is NEGATIVE;
            // no line at all (or flat alpha_moved) is VOID.
            eprintln!(
                "[interm-alpha] layer={li} op={relu_op} pairs={} extra_tightened={ia_tightened} \
extra_stabilized={ia_stabilized} alpha_moved={:?} ia_secs={:.2}",
                self.ia_pass_pairs, self.ia_alpha_moved, self.ia_secs
            );
        }
        eprintln!(
            "[backward-interm] layer={li} op={relu_op} unst={unst_before} cols={done}/{} \
tightened={tightened} stabilized={stabilized} secs={:.2} total_secs={:.2}{}",
            cols.len(),
            layer_t0.elapsed().as_secs_f64(),
            self.t0.elapsed().as_secs_f64(),
            if self.exhausted {
                " BUDGET-EXHAUSTED"
            } else {
                ""
            }
        );
    }

    /// #router-r3 `rule.ia_yield`: when the acting optimizer's candidate
    /// governor fired AND the phase then ended before its own budget, the
    /// unspent phase time is credited to the ascent's CONTRACT (excluded from
    /// the ascent's window, so it reaches the tree) and recorded as a yield the
    /// tree receives at its start. A no-op unless the stop authority armed
    /// `rule.ia_yield` (every other run: nothing is read or printed).
    fn hand_on_ia_yield(&self) {
        let Some(layer) = self.ia_yield_fired_layer else {
            return;
        };
        let authority = ny_core::stop_authority::global();
        if authority.ia_yield().is_none() {
            return;
        }
        let used = self.t0.elapsed().as_secs_f64();
        let unspent = if self.exhausted {
            0.0
        } else {
            (self.budget - used).max(0.0)
        };
        let ms = |secs: f64| (secs.max(0.0) * 1e3).min(u64::MAX as f64) as u64;
        authority.credit_ascent(std::time::Duration::from_secs_f64(unspent.min(1e9)));
        authority.record_yield(ny_core::stop_authority::YieldEvent {
            site: ny_core::stop_authority::StopSite::IntermAlpha,
            tag: "[interm-alpha]".to_owned(),
            unit: layer as u64,
            budget_ms: ms(self.budget),
            used_ms: ms(used),
            freed_ms: ms(unspent),
            rate: self.ia_yield_fired_rate,
            best: self.ia_yield.best(),
            floor: self.ia_yield.floor(),
        });
        eprintln!(
            "[optimizer] yield rule=rule.ia_yield site=margin-row.backward_interm layer={layer} \
             budget_ms={} used_ms={} freed_ms={} exhausted={} best={:.1} floor={:.1} \
             credited_to=ascent_contract",
            ms(self.budget),
            ms(used),
            ms(unspent),
            u8::from(self.exhausted),
            self.ia_yield.best(),
            self.ia_yield.floor()
        );
    }

    /// Final summary line (engagement telemetry for the whole build).
    pub(crate) fn finish(&self) -> BackwardIntermExecutionEvidence {
        self.hand_on_ia_yield();
        if self.enabled && self.cfg.interm_alpha > 0 {
            eprintln!(
                "[interm-alpha] done: dose={} pairs={} alpha_moved_per_policy={:?} \
extra_tightened={} extra_stabilized={} pass_errors={} ia_secs={:.2}",
                self.cfg.interm_alpha,
                self.ia_pass_pairs,
                self.ia_alpha_moved,
                self.ia_tightened_total,
                self.ia_stabilized_total,
                self.ia_pass_errors,
                self.ia_secs
            );
        }
        if self.enabled {
            eprintln!(
                "[backward-interm] done: chunks_declined={} last_chunk={:.2}s tightened={} \
stabilized={} crossed={} pass_errors={} \
work_secs={:.2} secs={:.2} exhausted={}",
                self.chunks_declined,
                self.last_chunk_secs,
                self.tightened_total,
                self.stabilized_total,
                self.crossed_total,
                self.pass_errors,
                self.work_secs,
                self.t0.elapsed().as_secs_f64(),
                self.exhausted
            );
        }
        if self.enabled && self.window {
            // #bi-window's OWN observable, unconditional when armed: `pairs=0`
            // with `fallbacks>0` means the crop never engaged (VOID for this
            // lever, not negative); `row_ratio` is the walked-row fraction the
            // cropped pairs kept.
            eprintln!(
                "[bi-window] done: pairs={} fallbacks={} rows_window={} rows_dense={} \
row_ratio={:.4}",
                self.win_pairs,
                self.win_fallbacks,
                self.win_rows_window,
                self.win_rows_dense,
                if self.win_rows_dense > 0 {
                    self.win_rows_window as f64 / self.win_rows_dense as f64
                } else {
                    0.0
                }
            );
        }
        if self.enabled && self.patch.is_some() {
            // #bi-patch's OWN observable, unconditional when armed: `pairs=0`
            // means patch mode never engaged (VOID for this lever, not
            // negative); `macs_patch`/`macs_dense` is the signed conv work the
            // patch pairs did against the dense pairs they replaced.
            eprintln!(
                "[bi-patch] done: pairs={} fused_cols={} fallbacks={} converted={} \
macs_patch={:.4e} macs_dense={:.4e} rows_patch={} rows_dense={} secs={:.2}",
                self.patch_pairs,
                self.patch_fused_cols,
                self.patch_fallbacks,
                self.patch_converted,
                self.patch_macs_patch,
                self.patch_macs_dense,
                self.patch_rows_patch,
                self.patch_rows_dense,
                self.patch_secs
            );
        }
        BackwardIntermExecutionEvidence {
            measured_work_secs: self.work_secs,
            enabled: self.enabled,
            configured_secs: self.cfg.secs,
            configured_chunk: self.cfg.chunk,
            configured_chunk_secs: self.cfg.chunk_secs,
            effective_chunk_secs: self.effective_chunk_secs,
            configured_topk: self.cfg.topk,
            eligible_layers: self.eligible_layers,
            suppressed_setup_layers: self.suppressed_setup_layers,
            layers_started: self.layers_started,
            chunks_completed: self.chunks_completed,
            columns_completed: self.columns_completed,
            deadline_capped: self.deadline_capped,
            budget_exhausted: self.exhausted,
            chunk_secs_capped: self.chunk_secs_capped,
            chunk_counterfactual_chunks: self.chunk_counterfactual_chunks,
            chunk_ceiling_binding_chunks: self.chunk_ceiling_binding_chunks,
            fixed_width_chunks: self.fixed_width_chunks,
            adaptive_width_changes: self.adaptive_width_changes,
            topk_binding_layers: self.topk_binding_layers,
            policy_mismatch: false,
        }
    }
}

/// #bi-window (cct plan P5 step 1): RECEPTIVE-FIELD-CROPPED prefix passes.
///
/// WHY. Every `#backward-interm` column is an identity row seeded at one trunk
/// pre-activation and walked back to the input. The engine walks it as a DENSE
/// full-tensor pass, so a column costs every row of every prefix tensor even
/// though its coefficients are EXACTLY zero outside the column's receptive
/// field. Measured on cct narrow CNN7 (`scratchpad/vbs/diag6/cct.md` §2a): the
/// plain per-column cost tracks the dense prefix MAC count (2.2 / 3.9-4.9 /
/// 6.3-6.7 / 7.3 ms per Lower+Upper column at relu_b..relu_e), so the phase
/// covers ~30% of the unstable neurons and never reaches trunk layer 4 inside
/// the 120 s tier. A crop removes the zero work.
///
/// WHAT. For a chunk of columns (one spatial tile of the seeded layer), build a
/// SMALLER NETWORK — the prefix ops restricted to the chunk's receptive-field
/// cone, one rectangle per tensor — plus the matching slice of the root gates
/// and the input box, and run the UNCHANGED certified engine on it.
///
/// SOUNDNESS (by construction; nothing in the engine changes).
/// * The cropped net computes EXACTLY the seeded pre-activations as a function
///   of the cropped input. Each conv crop is the same kernel/stride/weights over
///   the output rectangle, with the crop's own padding: a crop-coordinate tap is
///   either a real input inside the input rectangle or a position that was
///   ZERO PADDING in the full conv (`conv_input_rect` clamps only at the image
///   border), so every cropped output equals the full output at that position.
///   Relu / ChannelAffine are elementwise and are cropped to the same
///   rectangle. The chain rule restricted to the cone therefore reproduces the
///   full backward recursion on every row that can carry a nonzero coefficient.
/// * The pre-activations do not depend on inputs outside the cone, so bounds
///   over the CROPPED input box equal bounds over the full box.
/// * The engine instantiates every certified envelope at the CROPPED net's own
///   sizes (`gamma_n(n + ..)` per op, `k_bwd` per conv, `gamma_n(n_in + 16)`
///   in `concretize`). Those sizes are the true accumulation lengths of the
///   nonzero terms: the rows the crop drops contribute exact `±0.0` in the
///   dense pass (an f64 `x + 0.0` rounds nothing), and so do their error
///   entries in exact arithmetic (the dense pass carries `next_up(0)`
///   subnormals there only because `certify_up(0)` is conservative). The crop is
///   thus a valid certified pass of its own, and it is never LOOSER than the
///   dense pass: every coefficient and bias it keeps is the same f64 value (up
///   to the sign of zero), and every error entry is dominated entrywise by the
///   dense one because each error update is monotone in its inputs.
///   Pinned by `bi_window_matches_dense_and_is_never_looser` and
///   `bi_window_bounds_contain_sampled_preactivations`.
/// * FAIL-CLOSED: anything the crop cannot express — a residual `Add`, a
///   `Flatten`, a transposed conv, a tensor with two consumers, a shape the
///   conv geometry does not pin, an out-of-range column — returns `None` and the
///   caller runs the dense pass. The crop is an execution strategy, never a
///   relaxation. The STRUCTURAL refusals are made once per layer, at
///   `prefix_shapes`, so a layer the crop can never express is scheduled
///   exactly as the dark phase schedules it (no tile-major cap on its dense
///   chunks); only geometry refusals are left to the per-chunk
///   `CropPlan::build`.
pub(crate) mod window {
    use ndarray::Array2;
    use ny_core::Result;

    use super::super::engine::{DomainGates, GateVecs, LaneDir, Seed};
    use super::super::gpu_conv::GpuConvRoute;
    use super::super::net::{compile_conv, ConvOp, TwinNet, TwinOp};
    use super::super::root::{LayerGates, RootGates};

    /// `(C, H, W)` of one tensor.
    pub(crate) type Shape = (usize, usize, usize);

    /// Half-open spatial rectangle `[y0, y1) x [x0, x1)`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct Rect {
        pub(crate) y0: usize,
        pub(crate) y1: usize,
        pub(crate) x0: usize,
        pub(crate) x1: usize,
    }

    impl Rect {
        fn h(self) -> usize {
            self.y1 - self.y0
        }

        fn w(self) -> usize {
            self.x1 - self.x0
        }
    }

    /// One cropped Lower+Upper pair, concretized over the cropped input box.
    pub(crate) struct WindowedPair {
        /// Certified lower bound per requested column (request order).
        pub(crate) lb: Vec<f64>,
        /// Certified upper bound per requested column (request order).
        pub(crate) ub: Vec<f64>,
        /// Rows of the tensors the cropped walk visits.
        pub(crate) rows_window: usize,
        /// Rows of the same tensors in the dense walk.
        pub(crate) rows_dense: usize,
    }

    fn set_shape(shapes: &mut [Option<Shape>], tensor: usize, shape: Shape) -> bool {
        match shapes.get(tensor).copied() {
            Some(None) => {
                shapes[tensor] = Some(shape);
                true
            }
            Some(Some(previous)) => previous == shape,
            None => false,
        }
    }

    /// THE CROP'S STRUCTURAL ADMISSION, shared by `prefix_shapes` (the phase's
    /// once-per-layer gate) and `CropPlan::build` (once per chunk) so the two
    /// can never disagree: `relu_op` is a trunk ReLU strictly before the head,
    /// and the prefix `ops[..=relu_op]` is a single-consumer chain of
    /// non-transposed Conv / ChannelAffine / Relu ops over tensors
    /// `0..=relu_op + 1`.
    ///
    /// A residual `Add` fails it EVEN WHEN every tensor's shape is pinned: a
    /// conv downstream of the block pins the block's relu output, and that
    /// shape-preserving relu back-fills the `Add` output (every ResNet trunk
    /// layer past its first block). Before this gate, `prefix_shapes` answered
    /// `Some` there, so the phase tile-major-ordered the layer and capped every
    /// dense chunk at one tile run — for a crop `CropPlan::build` refuses on
    /// every chunk. Refusing here runs such a layer exactly as the dark phase
    /// (historical order and widths, one fallback per layer). Scheduling only:
    /// a refusal always means the unchanged dense pair.
    fn chain_prefix(net: &TwinNet, relu_op: usize) -> bool {
        if relu_op >= net.i_gemm1 || relu_op >= net.ops.len() {
            return false;
        }
        if !matches!(net.ops[relu_op], TwinOp::Relu { .. }) {
            return false;
        }
        let mut consumed = vec![false; relu_op + 2];
        for op in &net.ops[..=relu_op] {
            let input = match op {
                TwinOp::Conv(cv) if !cv.transposed => cv.input,
                TwinOp::Relu { input, .. } | TwinOp::ChannelAffine { input, .. } => *input,
                _ => return false,
            };
            match consumed.get_mut(input) {
                Some(seen) if !*seen => *seen = true,
                _ => return false,
            }
        }
        true
    }

    /// `(C, H, W)` of every tensor `0..=relu_op + 1`, pinned by the conv
    /// geometry and carried through the shape-preserving ops. `None` when the
    /// prefix is not a croppable chain (`chain_prefix`: a residual `Add`, a
    /// `Flatten`, a transposed conv, a tensor with two consumers, a seed that
    /// is not a trunk relu), when any tensor stays unpinned, two pins
    /// disagree, or a pin contradicts the compiled flat size.
    ///
    /// `Some` is the phase's per-layer licence to schedule the layer
    /// tile-major; `CropPlan::build` can still refuse a chunk on its
    /// GEOMETRY (the seed rectangle, the crop padding), never on structure.
    pub(crate) fn prefix_shapes(net: &TwinNet, relu_op: usize) -> Option<Vec<Shape>> {
        if !chain_prefix(net, relu_op) || relu_op + 1 >= net.tsize.len() {
            return None;
        }
        let tensors = relu_op + 2;
        let mut shapes: Vec<Option<Shape>> = vec![None; tensors];
        for (k, op) in net.ops[..=relu_op].iter().enumerate() {
            if let TwinOp::Conv(cv) = op {
                if !set_shape(&mut shapes, cv.input, cv.ishape)
                    || !set_shape(&mut shapes, k + 1, cv.oshape)
                {
                    return None;
                }
            }
        }
        for _ in 0..tensors {
            let mut changed = false;
            for (k, op) in net.ops[..=relu_op].iter().enumerate() {
                let input = match op {
                    TwinOp::Relu { input, .. } | TwinOp::ChannelAffine { input, .. } => *input,
                    _ => continue,
                };
                if input >= tensors {
                    return None;
                }
                match (shapes[input], shapes[k + 1]) {
                    (Some(a), None) => {
                        shapes[k + 1] = Some(a);
                        changed = true;
                    }
                    (None, Some(b)) => {
                        shapes[input] = Some(b);
                        changed = true;
                    }
                    (Some(a), Some(b)) if a != b => return None,
                    _ => {}
                }
            }
            if !changed {
                break;
            }
        }
        let shapes: Vec<Shape> = shapes.into_iter().collect::<Option<Vec<_>>>()?;
        for (t, &(c, h, w)) in shapes.iter().enumerate() {
            if c.checked_mul(h)?.checked_mul(w)? != net.tsize[t] {
                return None;
            }
        }
        Some(shapes)
    }

    /// The input rectangle a conv reads to produce `out`: every tap of every
    /// output in the rectangle, clamped ONLY at the image border (positions
    /// beyond it are the conv's zero padding). `None` when the rectangle is
    /// empty or reads no real input at all.
    pub(crate) fn conv_input_rect(cv: &ConvOp, out: Rect) -> Option<Rect> {
        if out.y1 <= out.y0 || out.x1 <= out.x0 {
            return None;
        }
        let (_, h, w) = cv.ishape;
        let (_, _, kh, kw) = cv.kernel;
        let (sh, sw) = cv.stride;
        let (pt, pl, _, _) = cv.pads;
        let span = |o0: usize, o1: usize, s: usize, k: usize, pad: usize, n: usize| {
            let start = (o0 * s) as isize - pad as isize;
            let end = ((o1 - 1) * s + k) as isize - pad as isize;
            let lo = start.max(0);
            let hi = end.min(n as isize);
            (hi > lo).then_some((lo as usize, hi as usize))
        };
        let (y0, y1) = span(out.y0, out.y1, sh, kh, pt, h)?;
        let (x0, x1) = span(out.x0, out.x1, sw, kw, pl, w)?;
        Some(Rect { y0, y1, x0, x1 })
    }

    /// Bounding rectangle of the columns' spatial positions in `shape`.
    pub(crate) fn rect_of(cols: &[usize], shape: Shape) -> Option<Rect> {
        let (c, h, w) = shape;
        let hw = h.checked_mul(w)?;
        let n = c.checked_mul(hw)?;
        if cols.is_empty() || hw == 0 {
            return None;
        }
        let (mut y0, mut y1, mut x0, mut x1) = (usize::MAX, 0usize, usize::MAX, 0usize);
        for &j in cols {
            if j >= n {
                return None;
            }
            let sp = j % hw;
            let (y, x) = (sp / w, sp % w);
            y0 = y0.min(y);
            y1 = y1.max(y + 1);
            x0 = x0.min(x);
            x1 = x1.max(x + 1);
        }
        Some(Rect { y0, y1, x0, x1 })
    }

    /// TILE-MAJOR order of `cols` (flat indices in `shape`) and, per position,
    /// the exclusive end of that position's tile run. The tile edge is chosen
    /// so a tile holds about `target` of the columns on average. Scheduling
    /// only: the same columns are bounded, just grouped differently.
    pub(crate) fn tile_order(
        cols: &[usize],
        shape: Shape,
        target: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let (_, h, w) = shape;
        let hw = h * w;
        if cols.is_empty() || hw == 0 || w == 0 {
            return (cols.to_vec(), vec![cols.len(); cols.len()]);
        }
        #[allow(clippy::cast_precision_loss)]
        let density = cols.len() as f64 / hw as f64;
        #[allow(clippy::cast_precision_loss)]
        let want = target.max(1) as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let edge = ((want / density.max(1e-12)).sqrt().ceil() as usize).clamp(1, h.max(w));
        let tiles_x = w.div_ceil(edge);
        let tile = |j: usize| {
            let sp = j % hw;
            ((sp / w) / edge) * tiles_x + (sp % w) / edge
        };
        let mut ordered = cols.to_vec();
        ordered.sort_by_key(|&j| (tile(j), j));
        let mut run_ends = vec![0usize; ordered.len()];
        let mut start = 0usize;
        while start < ordered.len() {
            let t = tile(ordered[start]);
            let mut end = start + 1;
            while end < ordered.len() && tile(ordered[end]) == t {
                end += 1;
            }
            for slot in &mut run_ends[start..end] {
                *slot = end;
            }
            start = end;
        }
        (ordered, run_ends)
    }

    /// The cropped prefix network for one seeded rectangle.
    pub(crate) struct CropPlan {
        net: TwinNet,
        /// Per tensor id: the FULL flat indices kept, ascending, in the cropped
        /// tensor's own channel-major order.
        keep: Vec<Vec<usize>>,
        /// Tensor the seed rows live on (the seeded relu's input).
        seed_tensor: usize,
        /// Flat input size of the FULL net (for the root-box slice check).
        full_n_in: usize,
        /// Flat size of every FULL tensor `0..=relu_op + 1`.
        full_sizes: Vec<usize>,
        rows_window: usize,
        rows_dense: usize,
    }

    impl CropPlan {
        /// Build the crop for seed rectangle `seed_rect` on the input of trunk
        /// relu op `relu_op`. `None` = not expressible (the caller goes dense).
        pub(crate) fn build(
            net: &TwinNet,
            relu_op: usize,
            shapes: &[Shape],
            seed_rect: Rect,
        ) -> Option<Self> {
            // A single-consumer chain of Conv / ChannelAffine / Relu ops only
            // (the same gate `prefix_shapes` applies once per layer; checked
            // again here so a caller holding forged shapes still fails closed).
            if !chain_prefix(net, relu_op) {
                return None;
            }
            let TwinOp::Relu {
                input: seed_tensor, ..
            } = &net.ops[relu_op]
            else {
                return None;
            };
            let seed_tensor = *seed_tensor;
            let tensors = relu_op + 2;
            if shapes.len() < tensors || seed_tensor >= tensors {
                return None;
            }
            let fits = |r: Rect, s: Shape| r.y0 < r.y1 && r.x0 < r.x1 && r.y1 <= s.1 && r.x1 <= s.2;
            if !fits(seed_rect, shapes[seed_tensor]) {
                return None;
            }
            // The receptive-field cone, one rectangle per tensor, walked from
            // the seed back to the input.
            let mut rects: Vec<Option<Rect>> = vec![None; tensors];
            rects[seed_tensor] = Some(seed_rect);
            rects[relu_op + 1] = Some(seed_rect);
            for k in (0..relu_op).rev() {
                let out = rects[k + 1]?;
                match &net.ops[k] {
                    TwinOp::Conv(cv) => {
                        let rin = conv_input_rect(cv, out)?;
                        if rects[cv.input].replace(rin).is_some() {
                            return None;
                        }
                    }
                    TwinOp::Relu { input, .. } | TwinOp::ChannelAffine { input, .. } => {
                        if rects[*input].replace(out).is_some() {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }
            let rects: Vec<Rect> = rects.into_iter().collect::<Option<Vec<_>>>()?;
            let mut keep: Vec<Vec<usize>> = Vec::with_capacity(tensors);
            for (t, &r) in rects.iter().enumerate() {
                let (c, h, w) = shapes[t];
                if !fits(r, (c, h, w)) {
                    return None;
                }
                let mut idx = Vec::with_capacity(c * r.h() * r.w());
                for ch in 0..c {
                    for y in r.y0..r.y1 {
                        for x in r.x0..r.x1 {
                            idx.push(ch * h * w + y * w + x);
                        }
                    }
                }
                keep.push(idx);
            }
            // Cropped ops, same indices and tensor ids as the full prefix.
            let mut ops = Vec::with_capacity(relu_op + 1);
            for (k, op) in net.ops[..=relu_op].iter().enumerate() {
                let cropped = match op {
                    TwinOp::Conv(cv) => {
                        let (rin, rout) = (rects[cv.input], rects[k + 1]);
                        let (_, _, kh, kw) = cv.kernel;
                        let (sh, sw) = cv.stride;
                        let (pt, pl, _, _) = cv.pads;
                        // Crop-coordinate padding: the first window starts at
                        // `-top`; the last window must fit `top + H' + bottom`.
                        let top = (rin.y0 + pt).checked_sub(rout.y0 * sh)?;
                        let left = (rin.x0 + pl).checked_sub(rout.x0 * sw)?;
                        let bottom = ((rout.h() - 1) * sh + kh).checked_sub(rin.h() + top)?;
                        let right = ((rout.w() - 1) * sw + kw).checked_sub(rin.w() + left)?;
                        TwinOp::Conv(Box::new(compile_conv(
                            cv.input,
                            &cv.wmat,
                            &cv.bias,
                            &cv.bias_err,
                            cv.weight_rel_err,
                            cv.kernel,
                            cv.stride,
                            (top, left, bottom, right),
                            (cv.ishape.0, rin.h(), rin.w()),
                            (cv.oshape.0, rout.h(), rout.w()),
                        )))
                    }
                    TwinOp::Relu { input, layer } => TwinOp::Relu {
                        input: *input,
                        layer: *layer,
                    },
                    TwinOp::ChannelAffine {
                        input,
                        scale,
                        shift,
                        scale_rel_err,
                        shift_err,
                    } => {
                        let kept = &keep[*input];
                        if scale.len() != net.tsize[*input]
                            || shift.len() != scale.len()
                            || shift_err.len() != scale.len()
                        {
                            return None;
                        }
                        TwinOp::ChannelAffine {
                            input: *input,
                            scale: kept.iter().map(|&i| scale[i]).collect(),
                            shift: kept.iter().map(|&i| shift[i]).collect(),
                            scale_rel_err: *scale_rel_err,
                            shift_err: kept.iter().map(|&i| shift_err[i]).collect(),
                        }
                    }
                    _ => return None,
                };
                ops.push(cropped);
            }
            let tsize: Vec<usize> = keep.iter().map(Vec::len).collect();
            // The walk visits tensors `0..=seed_tensor`.
            let rows_window: usize = tsize[..=seed_tensor].iter().sum();
            let rows_dense: usize = net.tsize[..=seed_tensor].iter().sum();
            let trunk_relus: Vec<usize> = net
                .trunk_relus
                .iter()
                .copied()
                .filter(|&op| op <= relu_op)
                .collect();
            Some(Self {
                net: TwinNet {
                    n_in: tsize[0],
                    ops,
                    tsize,
                    // Strictly past the seeded relu: the engine's prefix entry
                    // refuses a seed op at or beyond the head.
                    i_gemm1: relu_op + 1,
                    trunk_relus,
                    n_y: 0,
                    n_out: 0,
                },
                keep,
                seed_tensor,
                full_n_in: net.n_in,
                full_sizes: net.tsize[..tensors].to_vec(),
                rows_window,
                rows_dense,
            })
        }

        fn pick(kept: &[usize], values: &[f64]) -> Option<Vec<f64>> {
            kept.iter().map(|&i| values.get(i).copied()).collect()
        }

        /// The root restricted to the cone: prefix-layer gates on their kept
        /// neurons, and the input box on the kept inputs. Everything a prefix
        /// pass reads, nothing else (no clip rows, no head state, no memo).
        fn crop_root(&self, root: &RootGates, seed_layer: usize) -> Option<RootGates> {
            if root.mid.len() != self.full_n_in
                || root.rad.len() != self.full_n_in
                || root.xabs.len() != self.full_n_in
                || root.lo.len() != self.full_n_in
                || root.hi.len() != self.full_n_in
                || root.layers.len() < seed_layer
            {
                return None;
            }
            let kept_in = &self.keep[0];
            let mut layers = Vec::with_capacity(seed_layer);
            for li in 0..seed_layer {
                let op = *self.net.trunk_relus.get(li)?;
                let TwinOp::Relu { input, layer } = self.net.ops.get(op)? else {
                    return None;
                };
                let full = &root.layers[li];
                if *layer != li || full.n != self.keep_full_size(*input)? {
                    return None;
                }
                let kept = &self.keep[*input];
                let unst: Vec<usize> = full
                    .unst
                    .iter()
                    .filter_map(|&j| kept.binary_search(&j).ok())
                    .collect();
                layers.push(LayerGates {
                    op: full.op,
                    n: kept.len(),
                    l: Self::pick(kept, &full.l)?,
                    u: Self::pick(kept, &full.u)?,
                    alpha: Self::pick(kept, &full.alpha)?,
                    s: Self::pick(kept, &full.s)?,
                    c: Self::pick(kept, &full.c)?,
                    ms: Self::pick(kept, &full.ms)?,
                    unst,
                    clip_rows: None,
                });
            }
            Some(RootGates {
                mode: root.mode,
                mid: Self::pick(kept_in, &root.mid)?,
                rad: Self::pick(kept_in, &root.rad)?,
                xabs: Self::pick(kept_in, &root.xabs)?,
                layers,
                lo: Self::pick(kept_in, &root.lo)?,
                hi: Self::pick(kept_in, &root.hi)?,
                backward_interm_execution: super::BackwardIntermExecutionEvidence::default(),
                // Only the certified error lane is inherited: the crop builds
                // no tableau, so the precision flags stay at their defaults
                // (exactly the `tighten_layer` scratch root's rule).
                root_precision_policy: super::super::RootPrecisionRuntimePolicy {
                    err_lane: root.root_precision_policy.err_lane,
                    ..super::super::RootPrecisionRuntimePolicy::default()
                },
                root_precision_execution:
                    super::super::root::RootPrecisionExecutionEvidence::default(),
                spec_alpha_policy: None,
                head_alpha: None,
                root_eval_memo: super::super::bab::RootEvalMemo::default(),
                head_group_source: None,
            })
        }

        /// Full flat size of tensor `t` (the keep list's source tensor).
        fn keep_full_size(&self, t: usize) -> Option<usize> {
            self.full_sizes.get(t).copied()
        }

        /// A prefix gate override restricted to the cone. Split-Lagrangian
        /// terms are refused (no prefix pass of this phase carries them).
        fn crop_dom(&self, dom: &DomainGates, seed_layer: usize) -> Option<DomainGates> {
            if !dom.beta.is_empty() || !dom.beta_pc.is_empty() {
                return None;
            }
            let mut layers = std::collections::BTreeMap::new();
            for (&li, gv) in &dom.layers {
                if li >= seed_layer {
                    // Never read by a walk that stops before the seeded relu.
                    continue;
                }
                let op = *self.net.trunk_relus.get(li)?;
                let TwinOp::Relu { input, .. } = self.net.ops.get(op)? else {
                    return None;
                };
                let full_n = self.keep_full_size(*input)?;
                if gv.alpha.len() != full_n
                    || gv.s.len() != full_n
                    || gv.c.len() != full_n
                    || gv.ms.len() != full_n
                {
                    return None;
                }
                let kept = &self.keep[*input];
                layers.insert(
                    li,
                    GateVecs {
                        alpha: Self::pick(kept, &gv.alpha)?,
                        s: Self::pick(kept, &gv.s)?,
                        c: Self::pick(kept, &gv.c)?,
                        ms: Self::pick(kept, &gv.ms)?,
                    },
                );
            }
            Some(DomainGates {
                layers,
                infeasible: dom.infeasible,
                // #mr-close-telemetry: keep the emptiness flag and its
                // certificate's slack paired (telemetry only; never read by a
                // prefix walk).
                infeasible_slack: dom.infeasible_slack,
                ..DomainGates::default()
            })
        }

        /// Identity seed rows for `cols` (full flat indices on the seed
        /// tensor), in the crop's coordinates. Exact, so no error lane.
        fn seed(&self, cols: &[usize]) -> Option<Seed> {
            let kept = &self.keep[self.seed_tensor];
            let mut s = Array2::<f64>::zeros((kept.len(), cols.len()));
            for (c, &j) in cols.iter().enumerate() {
                let p = kept.binary_search(&j).ok()?;
                s[[p, c]] = 1.0;
            }
            Some(Seed { s, e: None })
        }
    }

    /// Certified bounds of `cols` (flat indices on the input of trunk relu op
    /// `relu_op`) from ONE cropped Lower+Upper pair under `root`'s prefix gates
    /// (optionally overridden by `dom`).
    ///
    /// `None`: the crop cannot express this request — the caller runs the dense
    /// pair. `Some(Err)`: the certified engine refused the cropped pass (the
    /// caller fails open exactly as for a dense refusal).
    ///
    /// `route` (#bi-gpu): the conv backward route of the cropped pair's engine;
    /// `None` is `BackwardEngine::new`'s own (the dark phase, byte-identical).
    pub(crate) fn prefix_bounds(
        net: &TwinNet,
        root: &RootGates,
        relu_op: usize,
        shapes: &[Shape],
        cols: &[usize],
        dom: Option<&DomainGates>,
        route: Option<&GpuConvRoute>,
    ) -> Option<Result<WindowedPair>> {
        let TwinOp::Relu { input, layer } = net.ops.get(relu_op)? else {
            return None;
        };
        let rect = rect_of(cols, *shapes.get(*input)?)?;
        let plan = CropPlan::build(net, relu_op, shapes, rect)?;
        let croot = plan.crop_root(root, *layer)?;
        let cdom = match dom {
            Some(d) => Some(plan.crop_dom(d, *layer)?),
            None => None,
        };
        let seed = plan.seed(cols)?;
        let eng = super::super::bi_gpu::engine(&plan.net, &croot, route);
        let lower = eng.run_prefix_dom(&seed, relu_op, LaneDir::Lower, cdom.as_ref());
        let upper = eng.run_prefix_dom(&seed, relu_op, LaneDir::Upper, cdom.as_ref());
        Some(match (lower, upper) {
            (Ok(lo), Ok(up)) => Ok(WindowedPair {
                lb: eng.concretize_lower(&lo),
                ub: eng.concretize_upper(&up),
                rows_window: plan.rows_window,
                rows_dense: plan.rows_dense,
            }),
            (Err(error), _) | (_, Err(error)) => Err(error),
        })
    }
}

#[cfg(test)]
mod parent_refresh_tests {
    use super::intersect_parent_refresh;

    #[test]
    fn crossed_interval_refuses_the_whole_incremental_frame() {
        assert_eq!(
            intersect_parent_refresh(-1.0, 0.25, -0.5, 0.5, 3, 7).unwrap(),
            (-0.5, 0.25)
        );
        let error = intersect_parent_refresh(-1.0, -0.25, 0.25, 1.0, 3, 7)
            .expect_err("a crossing must never return a partially refreshed frame");
        let message = error.to_string();
        assert!(message.contains("crossed at (3,7)"), "{message}");
    }
}

#[cfg(test)]
mod pass_sizer_admission_tests {
    use super::{
        chunk_ceiling_binds_width, chunk_width_differs_from_default, next_chunk_width,
        predicted_chunk_secs, BiState,
    };

    #[test]
    fn adaptive_admission_prices_the_width_about_to_run() {
        assert_eq!(next_chunk_width(2.0, None, 256, 1_000), 1);
        let width = next_chunk_width(2.0, Some(0.01), 256, 1_000);
        assert_eq!(width, 200);
        assert_eq!(predicted_chunk_secs(0.01, Some(0.01), width), Some(2.0));
    }

    #[test]
    fn a_narrow_tail_retains_the_measured_fixed_overhead() {
        assert_eq!(next_chunk_width(0.0, Some(0.01), 256, 7), 7);
        assert_eq!(
            predicted_chunk_secs(2.0, Some(0.01), 7),
            Some(2.0),
            "column scaling must not price a narrow affine-cost tail below the last pass"
        );
    }

    #[test]
    fn every_nondefault_search_chunk_can_change_a_live_width() {
        let remaining = 2_048;
        assert!(chunk_width_differs_from_default(0.0, None, 64, remaining));
        assert!(!chunk_width_differs_from_default(0.0, None, 256, remaining));
        assert!(chunk_width_differs_from_default(
            0.0, None, 1_024, remaining
        ));
        assert!(
            !chunk_width_differs_from_default(0.0, None, 1_024, 32),
            "a configured value earns no credit when the live workload is too narrow"
        );
    }

    #[test]
    fn a_default_chunk_needs_a_wider_live_workload_to_bind() {
        assert!(!chunk_ceiling_binds_width(0.0, None, 256, 36));
        assert!(!chunk_ceiling_binds_width(0.0, None, 256, 256));
        assert!(chunk_ceiling_binds_width(0.0, None, 256, 2_048));
        assert!(!chunk_ceiling_binds_width(2.0, None, 256, 2_048));
        assert!(chunk_ceiling_binds_width(
            2.0,
            Some(2.0 / 1_024.0),
            256,
            2_048,
        ));
    }

    #[test]
    fn adaptive_target_reports_cap_collisions_exactly() {
        let policy = |chunk_secs| super::super::BackwardIntermRuntimePolicy {
            enabled: true,
            secs: 20.0,
            chunk: 256,
            chunk_secs,
            topk: 1_024,
            interm_alpha: 0,
            window: false,
            gpu: false,
            patch: super::super::BiPatchRuntimePolicy::default(),
        };
        let exact = BiState::new(policy(2.0), None);
        assert_eq!(exact.effective_chunk_secs, 2.0);
        assert!(!exact.chunk_secs_capped);
        let capped = BiState::new(policy(5.0), None);
        assert_eq!(capped.effective_chunk_secs, 2.0);
        assert!(capped.chunk_secs_capped);

        let sub_floor = BiState::new(policy(5e-10), None);
        assert_eq!(sub_floor.effective_chunk_secs, 5e-10);
        assert!(!sub_floor.chunk_secs_capped);

        let mut zero_budget = BiState::new(
            super::super::BackwardIntermRuntimePolicy {
                secs: 0.0,
                chunk_secs: 2.0,
                ..policy(2.0)
            },
            None,
        );
        assert_eq!(zero_budget.effective_chunk_secs, 1e-9);
        assert!(zero_budget.chunk_secs_capped);
        assert!(zero_budget.out_of_time());
        assert!(zero_budget.exhausted);

        let disabled_zero = BiState::new(
            super::super::BackwardIntermRuntimePolicy {
                enabled: false,
                secs: 0.0,
                ..policy(2.0)
            },
            None,
        );
        assert!(!disabled_zero.would_enter_governed_setup_now());
    }
}

#[cfg(test)]
mod bi_yield_tests {
    use super::{BiCfg, BiState, IA_YIELD_MIN_CHARGE_SECS};

    /// Drive the governor directly against synthetic accounting. This tests the
    /// DECISION, not the bound work — the bound work is already covered by the
    /// phase's own suites and is unchanged by this commit.
    fn state(window: usize, ratio: f64) -> BiState {
        let mut st = BiState::new(
            super::super::BackwardIntermRuntimePolicy {
                enabled: true,
                ..Default::default()
            },
            None,
        );
        st.cfg = BiCfg {
            secs: 40.0,
            chunk: 64,
            chunk_secs: 0.0,
            topk: 20,
            yield_window: window,
            yield_ratio: ratio,
            interm_alpha: 0,
        };
        st.budget = 40.0;
        st
    }

    /// A window whose tightening rate collapses must stop the phase and hand the
    /// remaining budget back. This is the whole point: on cifar100 resnet_large
    /// the phase spends its 40 s cap and leaves BaB 0-40 expansions.
    #[test]
    fn collapsing_marginal_yield_stops_the_phase() {
        let mut st = state(2, 0.25);
        // Window 1: a high rate establishes the baseline (100 tightened / 1 s).
        st.yield_layers_done = 2;
        st.tightened_total = 100;
        st.work_secs = 1.0;
        assert!(
            !st.yield_collapsed(),
            "the first window sets the baseline, it must never stop"
        );
        // Window 2: 1 tightened in another second — a 100x collapse.
        st.yield_layers_done = 4;
        st.tightened_total = 101;
        st.work_secs = 2.0;
        assert!(
            st.yield_collapsed(),
            "a window at 1% of the best rate must stop the phase; otherwise it spends \
             its whole cap earning nothing, which is the measured resnet_large behaviour"
        );
        assert!(
            st.yield_stopped,
            "the stop must be attributable to the yield check"
        );
    }

    /// A phase that keeps earning must NOT be stopped — the governor must not
    /// cost tightness on the rows where the phase is doing its job (measured:
    /// resnet_medium finishes in 8.0-8.2 s without ever binding the cap).
    #[test]
    fn a_still_productive_phase_is_not_stopped() {
        let mut st = state(2, 0.25);
        st.yield_layers_done = 2;
        st.tightened_total = 100;
        st.work_secs = 1.0;
        assert!(!st.yield_collapsed());
        // Window 2 at 80% of the best rate — well above the 25% floor.
        st.yield_layers_done = 4;
        st.tightened_total = 180;
        st.work_secs = 2.0;
        assert!(
            !st.yield_collapsed(),
            "80% of the best rate is still productive"
        );
        assert!(!st.yield_stopped);
    }

    /// `yield_window: 0` must be byte-identical to the historical phase: the
    /// check returns immediately and never mutates accounting state.
    #[test]
    fn yield_window_zero_is_inert() {
        let mut st = state(0, 0.25);
        st.yield_layers_done = 99;
        st.tightened_total = 0;
        st.work_secs = 39.0;
        assert!(!st.yield_collapsed(), "OFF must never stop the phase");
        assert!(!st.yield_stopped);
        assert_eq!(
            st.yield_last_chunks, 0,
            "OFF must not touch the window baseline"
        );
        assert_eq!(st.yield_best_rate, 0.0, "OFF must not record a rate");
    }

    /// Replay of the REAL measured layer rates from cifar100 resnet_large
    /// idx_1280_sidx_5310 (2026-09-16, arm binary `648a3aff1c69b90a`):
    ///
    /// ```text
    /// layer=1 op=3  tightened=1024 secs=6.37   -> 161 /s
    /// layer=3 op=11 tightened=301  secs=8.32   ->  36 /s   (0.22x best)
    /// layer=5 op=20 tightened=219  secs=10.08  ->  22 /s   (0.14x best)
    /// layer=7 op=29 tightened=0    secs=0.00   BUDGET-EXHAUSTED
    /// ```
    ///
    /// The chunk-keyed window never fired on this data — the bug this test
    /// exists to prevent. Layer-keyed, it must stop after layer 2, leaving the
    /// ~17 s that layers 3 and 4 consumed to the tree search.
    #[test]
    fn the_measured_resnet_large_layer_rates_stop_the_phase() {
        let mut st = state(1, 0.25);
        // layer 1: 1024 tightened in 6.37 s -> the baseline.
        st.yield_layers_done = 1;
        st.tightened_total = 1024;
        st.work_secs = 6.37;
        assert!(!st.yield_collapsed(), "the first layer sets the baseline");
        // layer 2: 301 more in 8.32 s -> 36/s, 0.22x the best.
        st.yield_layers_done = 2;
        st.tightened_total = 1024 + 301;
        st.work_secs = 6.37 + 8.32;
        assert!(
            st.yield_collapsed(),
            "36/s against a 161/s baseline is 0.22x — below the 0.25 floor — so the \
             phase must stop here instead of spending 17 more seconds for 219 more \
             tightenings and then exhausting its budget"
        );
        assert!(st.yield_stopped);
    }

    /// THE BUG THIS PINS (made and caught 2026-09-16, twice): a window sized at
    /// or above the phase's layer count is INERT — the first evaluation lands on
    /// the final layer, so the baseline is set and never compared. Both failed
    /// arms had exactly this shape: `yield_window: 4` against a phase that
    /// completes 4 layers, `yield_stops=0`, root unchanged at 35.1-35.8 s. The
    /// governor read as a negative result and was not one.
    ///
    /// The phase's layer count is a property of the NETWORK, not of config, so
    /// this cannot be asserted statically — hence a test that pins the failure
    /// mode and a comment at the arming site naming the constraint.
    #[test]
    fn a_window_at_the_layer_count_is_inert() {
        let mut st = state(4, 0.25);
        // Replay the measured four layers; a window of 4 must never stop.
        for (layer, tight, secs) in [
            (1usize, 1024usize, 6.77f64),
            (2, 1325, 15.09),
            (3, 1544, 25.51),
            (4, 1544, 25.51),
        ] {
            st.yield_layers_done = layer;
            st.tightened_total = tight;
            st.work_secs = secs;
            assert!(
                !st.yield_collapsed(),
                "window=4 over a 4-layer phase must stay inert at layer {layer}; \
                 firing here would mean the window arithmetic changed"
            );
        }
        assert!(
            !st.yield_stopped,
            "an inert window must never mark the phase stopped"
        );
        // The same data with window=1 MUST stop — proving the data is not the
        // reason the window-4 case stayed quiet.
        let mut st1 = state(1, 0.25);
        st1.yield_layers_done = 1;
        st1.tightened_total = 1024;
        st1.work_secs = 6.77;
        assert!(!st1.yield_collapsed(), "layer 1 is the baseline");
        st1.yield_layers_done = 2;
        st1.tightened_total = 1325;
        st1.work_secs = 15.09;
        assert!(
            st1.yield_collapsed(),
            "with window=1 the SAME measured data must stop at layer 2 (36/s vs a \
             151/s baseline = 0.24 < 0.25)"
        );
    }

    /// A rate of exactly the ratio must NOT stop (strict inequality), so the
    /// threshold is a floor rather than a coin-flip at the boundary.
    #[test]
    fn exactly_at_the_ratio_does_not_stop() {
        let mut st = state(1, 0.5);
        st.yield_layers_done = 1;
        st.tightened_total = 100;
        st.work_secs = 1.0;
        assert!(!st.yield_collapsed());
        st.yield_layers_done = 2;
        st.tightened_total = 150;
        st.work_secs = 2.0; // rate 50 == 0.5 * 100
        assert!(!st.yield_collapsed(), "boundary must not trip the stop");
    }

    // ---------------------------------------------------------------------
    //  #ia-yield -- the candidate-loop governor, NOT the phase latch
    // ---------------------------------------------------------------------

    fn ia_state(interm_alpha: usize) -> BiState {
        BiState::new(
            super::super::BackwardIntermRuntimePolicy {
                enabled: true,
                secs: 40.0,
                chunk: 256,
                chunk_secs: 0.0,
                topk: 1_024,
                interm_alpha,
                window: false,
                gpu: false,
                patch: super::super::BiPatchRuntimePolicy::default(),
            },
            None,
        )
    }

    /// The whole point of a separate governor: firing it must leave the phase
    /// running. `#bi-yield` sets `exhausted` and is measured to cost a verdict;
    /// `#ia-yield` must touch only the per-start-node candidates.
    #[test]
    fn ia_yield_is_not_the_phase_latch() {
        let mut st = ia_state(2);
        assert!(!st.exhausted);
        assert!(!st.ia_yield_stopped);
        // Simulate the rule firing, exactly as the layer hook does.
        st.ia_yield_stopped = true;
        st.ia_yield_fired_layer = Some(3);
        assert!(
            !st.exhausted,
            "#ia-yield must never set the phase latch: the cheap baseline \
             shrink-only pass keeps running on later layers"
        );
        assert!(!st.yield_stopped, "#bi-yield's own flag must stay clear");
    }

    /// Shipped state: inert. The candidates run on every layer exactly as today.
    #[test]
    fn ia_yield_ships_off_on_a_fresh_state() {
        let st = ia_state(2);
        assert!(
            !st.ia_yield.armed(),
            "IA_GOVERNOR_RATIO is None, so the candidate loop is ungoverned"
        );
        assert!(!st.ia_yield_stopped);
        assert_eq!(st.ia_yield_fired_layer, None);
        assert_eq!(st.ia_yield_last_secs, 0.0);
    }

    /// The divide-by-zero guard. A layer that charged no candidate wall must not
    /// be priced: its rate would be 0 against the 1e-9 floor and would end the
    /// lever on a layer that never ran.
    #[test]
    fn ia_yield_ignores_a_layer_that_charged_no_candidate_wall() {
        const {
            assert!(
                IA_YIELD_MIN_CHARGE_SECS > 0.0,
                "a zero floor is the divide-by-zero bug this constant exists to stop"
            )
        };
        let st = ia_state(2);
        // Fresh state: ia_secs == ia_yield_last_secs == 0, so the delta is 0.0
        // and is strictly below the floor -- the hook must not observe.
        assert!(
            (st.ia_secs - st.ia_yield_last_secs).partial_cmp(&IA_YIELD_MIN_CHARGE_SECS)
                != Some(std::cmp::Ordering::Greater),
            "an unworked layer must be skipped, not priced at rate 0"
        );
    }

    /// A dose of 0 skips the candidate block entirely, so the governor must
    /// never observe -- otherwise a byte-identical path would grow an observable.
    #[test]
    fn ia_yield_never_observes_at_dose_zero() {
        let st = ia_state(0);
        assert_eq!(st.cfg.interm_alpha, 0);
        assert_eq!(
            st.ia_yield.samples(),
            0,
            "dose 0 is byte-identical to the pre-#interm-alpha phase"
        );
    }
}

#[cfg(test)]
pub(crate) mod window_tests {
    //! #bi-window soundness pins: the cropped pair is a certified pass that is
    //! never looser than the dense pair, contains every sampled pre-activation,
    //! refuses what it cannot express, and is dark by default.

    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use super::super::engine::{BackwardEngine, LaneDir, Seed};
    use super::super::net::{TwinNet, TwinOp};
    use super::super::root::{LayerGates, RootGates};
    use super::super::rounding::RoundMode;
    use super::super::spec::{TwinOpSpec, TwinSpec};
    use super::{window, BiState, WINDOW_TARGET_COLS};

    /// Conv chain with an input normalization: 2x8x8 -> affine -> conv 4x8x8
    /// -> relu -> conv(s2) 4x4x4 -> relu -> conv 3x4x4 -> relu -> head.
    /// Every trunk tensor is spatial, so every crop below is strict.
    pub(crate) fn chain_spec(rng: &mut StdRng, scale: f64) -> TwinSpec {
        let mut w =
            |n: usize| -> Vec<f64> { (0..n).map(|_| rng.random_range(-scale..scale)).collect() };
        let conv = |input: usize,
                    weight: Vec<f64>,
                    bias: Vec<f64>,
                    kernel: (usize, usize, usize, usize),
                    stride: (usize, usize),
                    ishape: (usize, usize, usize),
                    oshape: (usize, usize, usize)| TwinOpSpec::Conv {
            input,
            weight,
            bias,
            bias_err: vec![1e-18; kernel.0],
            weight_rel_err: 1e-15,
            kernel,
            stride,
            pads: (1, 1, 1, 1),
            ishape,
            oshape,
        };
        let c1 = conv(
            1,
            w(4 * 2 * 9),
            w(4),
            (4, 2, 3, 3),
            (1, 1),
            (2, 8, 8),
            (4, 8, 8),
        );
        let c2 = conv(
            3,
            w(4 * 4 * 9),
            w(4),
            (4, 4, 3, 3),
            (2, 2),
            (4, 8, 8),
            (4, 4, 4),
        );
        let c3 = conv(
            5,
            w(3 * 4 * 9),
            w(3),
            (3, 4, 3, 3),
            (1, 1),
            (4, 4, 4),
            (3, 4, 4),
        );
        let g1 = w(5 * 48);
        let b1 = w(5);
        let g2 = w(3 * 5);
        let b2 = w(3);
        TwinSpec {
            n_in: 128,
            ops: vec![
                TwinOpSpec::ChannelAffine {
                    input: 0,
                    scale: vec![1.1, 0.9],
                    shift: vec![0.05, -0.02],
                    scale_rel_err: 1e-16,
                    shift_err: vec![1e-18, 1e-18],
                    shape: (2, 8, 8),
                }, // t1
                c1,                               // t2
                TwinOpSpec::Relu { input: 2 },    // t3 (trunk relu 0)
                c2,                               // t4
                TwinOpSpec::Relu { input: 4 },    // t5 (trunk relu 1)
                c3,                               // t6
                TwinOpSpec::Relu { input: 6 },    // t7 (trunk relu 2)
                TwinOpSpec::Flatten { input: 7 }, // t8
                TwinOpSpec::Gemm {
                    input: 8,
                    weight: g1,
                    bias: b1,
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
                    shape: (5, 48),
                }, // t9
                TwinOpSpec::Relu { input: 9 },    // t10 (head relu)
                TwinOpSpec::Gemm {
                    input: 10,
                    weight: g2,
                    bias: b2,
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
                    shape: (3, 5),
                }, // t11
            ],
        }
    }

    fn dense_pair(
        net: &TwinNet,
        root: &RootGates,
        relu_op: usize,
        n_pre: usize,
        cols: &[usize],
    ) -> (Vec<f64>, Vec<f64>) {
        let mut s = Array2::<f64>::zeros((n_pre, cols.len()));
        for (c, &j) in cols.iter().enumerate() {
            s[[j, c]] = 1.0;
        }
        let seed = Seed { s, e: None };
        let eng = BackwardEngine::new(net, root);
        let lower = eng
            .run_prefix(&seed, relu_op, LaneDir::Lower)
            .expect("dense lower");
        let upper = eng
            .run_prefix(&seed, relu_op, LaneDir::Upper)
            .expect("dense upper");
        (eng.concretize_lower(&lower), eng.concretize_upper(&upper))
    }

    fn seed_shape(net: &TwinNet, relu_op: usize, shapes: &[window::Shape]) -> window::Shape {
        let TwinOp::Relu { input, .. } = &net.ops[relu_op] else {
            panic!("op {relu_op} is not a relu");
        };
        shapes[*input]
    }

    /// NEVER LOOSER, AND EQUAL UP TO ROUNDING. Every coefficient and bias the
    /// crop keeps is the dense pass's value; every error entry is dominated,
    /// so the cropped lower bound is >= the dense one and the upper <=, both
    /// within rounding. A whole-layer request crops nothing and is identical.
    #[test]
    fn bi_window_matches_dense_and_is_never_looser() {
        let mut rng = StdRng::seed_from_u64(0xB1_3D01);
        let spec = chain_spec(&mut rng, 0.5);
        let net = TwinNet::compile(&spec).expect("compile");
        let (lo, hi) = (vec![-0.3; 128], vec![0.4; 128]);
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let mut strict = 0usize;
        for layer in root.layers.iter().skip(1) {
            let relu_op = layer.op;
            let shapes = window::prefix_shapes(&net, relu_op).expect("chain prefix is croppable");
            let (c, h, w) = seed_shape(&net, relu_op, &shapes);
            let n = c * h * w;
            assert_eq!(n, layer.n);
            let tile: Vec<usize> = (0..n)
                .filter(|&j| (j % (h * w)) / w < 2 && j % w < 2)
                .collect();
            for cols in [
                vec![0usize],
                vec![n - 1],
                vec![n / 2],
                tile,
                (0..n).collect(),
            ] {
                let pair = window::prefix_bounds(&net, &root, relu_op, &shapes, &cols, None, None)
                    .expect("a chain chunk is expressible")
                    .expect("the cropped pair is certified");
                let (dl, du) = dense_pair(&net, &root, relu_op, n, &cols);
                assert!(pair.rows_window <= pair.rows_dense);
                strict += usize::from(pair.rows_window < pair.rows_dense);
                for k in 0..cols.len() {
                    assert!(
                        pair.lb[k] >= dl[k] && pair.ub[k] <= du[k],
                        "op {relu_op} col {}: crop [{}, {}] is LOOSER than dense [{}, {}]",
                        cols[k],
                        pair.lb[k],
                        pair.ub[k],
                        dl[k],
                        du[k]
                    );
                    let tol = 1e-9 * (1.0 + dl[k].abs().max(du[k].abs()));
                    assert!(
                        pair.lb[k] - dl[k] <= tol && du[k] - pair.ub[k] <= tol,
                        "op {relu_op} col {}: crop drifted beyond rounding from dense",
                        cols[k]
                    );
                }
                if cols.len() == n {
                    assert_eq!(
                        pair.rows_window, pair.rows_dense,
                        "a whole layer crops nothing"
                    );
                    assert_eq!(pair.lb, dl, "an uncropped request must be bit-identical");
                    assert_eq!(pair.ub, du, "an uncropped request must be bit-identical");
                }
            }
        }
        assert!(
            strict > 0,
            "the fixture must exercise a STRICT crop (VOID otherwise)"
        );
    }

    /// SAMPLED CONTAINMENT: every cropped bound encloses the pre-activation at
    /// every sampled box point, for every neuron of every layer >= 1, chunked
    /// tile-major exactly as the phase chunks them.
    #[test]
    fn bi_window_bounds_contain_sampled_preactivations() {
        let mut rng = StdRng::seed_from_u64(0xB1_3D02);
        let spec = chain_spec(&mut rng, 0.6);
        let net = TwinNet::compile(&spec).expect("compile");
        let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let mut x = Array2::<f64>::zeros((128, 256));
        for i in 0..128 {
            for b in 0..256 {
                // Corners and interior points both.
                let u: f64 = if b % 4 == 0 {
                    if rng.random_range(0.0..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    }
                } else {
                    rng.random_range(-1.0..1.0)
                };
                x[[i, b]] = root.mid[i] + u * root.rad[i];
            }
        }
        let sel: std::collections::BTreeMap<usize, Vec<usize>> = root
            .layers
            .iter()
            .map(|layer| (layer.op, (0..layer.n).collect()))
            .collect();
        let (_, pre) = net.forward_points(&x, &sel).expect("forward");
        let mut checked = 0usize;
        for layer in root.layers.iter().skip(1) {
            let relu_op = layer.op;
            let shapes = window::prefix_shapes(&net, relu_op).expect("croppable");
            let all: Vec<usize> = (0..layer.n).collect();
            let (ordered, ends) = window::tile_order(
                &all,
                seed_shape(&net, relu_op, &shapes),
                WINDOW_TARGET_COLS / 16,
            );
            let mut cursor = 0usize;
            while cursor < ordered.len() {
                let part = &ordered[cursor..ends[cursor]];
                let pair = window::prefix_bounds(&net, &root, relu_op, &shapes, part, None, None)
                    .expect("expressible")
                    .expect("certified");
                let p = &pre[&relu_op];
                for (k, &j) in part.iter().enumerate() {
                    for b in 0..x.ncols() {
                        let v = p[[j, b]];
                        assert!(
                            v >= pair.lb[k] - 1e-9 && v <= pair.ub[k] + 1e-9,
                            "op {relu_op} neuron {j}: sampled {v} outside [{}, {}]",
                            pair.lb[k],
                            pair.ub[k]
                        );
                        checked += 1;
                    }
                }
                cursor = ends[cursor];
            }
        }
        assert!(
            checked > 10_000,
            "containment must be exercised, got {checked}"
        );
    }

    /// FAIL-CLOSED: a residual `Add` in the prefix, and an out-of-range column,
    /// are refused (`None`), so the caller runs the dense pass.
    #[test]
    fn bi_window_refuses_what_it_cannot_crop() {
        let mut rng = StdRng::seed_from_u64(0xB1_3D03);
        let mut w =
            |n: usize| -> Vec<f64> { (0..n).map(|_| rng.random_range(-0.4..0.4)).collect() };
        let residual = TwinSpec {
            n_in: 32,
            ops: vec![
                TwinOpSpec::Conv {
                    input: 0,
                    weight: w(2 * 2 * 9),
                    bias: w(2),
                    bias_err: vec![0.0; 2],
                    weight_rel_err: 1e-15,
                    kernel: (2, 2, 3, 3),
                    stride: (1, 1),
                    pads: (1, 1, 1, 1),
                    ishape: (2, 4, 4),
                    oshape: (2, 4, 4),
                }, // t1
                TwinOpSpec::Relu { input: 1 }, // t2
                TwinOpSpec::Conv {
                    input: 2,
                    weight: w(2 * 2 * 9),
                    bias: w(2),
                    bias_err: vec![0.0; 2],
                    weight_rel_err: 1e-15,
                    kernel: (2, 2, 3, 3),
                    stride: (1, 1),
                    pads: (1, 1, 1, 1),
                    ishape: (2, 4, 4),
                    oshape: (2, 4, 4),
                }, // t3
                TwinOpSpec::Add { lhs: 3, rhs: 2 }, // t4
                TwinOpSpec::Relu { input: 4 }, // t5
                TwinOpSpec::Flatten { input: 5 }, // t6
                TwinOpSpec::Gemm {
                    input: 6,
                    weight: w(3 * 32),
                    bias: w(3),
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
                    shape: (3, 32),
                }, // t7
                TwinOpSpec::Relu { input: 7 }, // t8
                TwinOpSpec::Gemm {
                    input: 8,
                    weight: w(2 * 3),
                    bias: w(2),
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
                    shape: (2, 3),
                }, // t9
            ],
        };
        let net = TwinNet::compile(&residual).expect("compile");
        let (lo, hi) = (vec![-0.2; 32], vec![0.3; 32]);
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let relu1 = root.layers[1].op;
        assert!(
            window::prefix_shapes(&net, relu1).is_none(),
            "the Add output is pinned by no conv, so the prefix is not croppable"
        );

        // The same residual block with a conv DOWNSTREAM of it (every ResNet
        // trunk layer past its first block): that conv pins the block's relu
        // output and the relu back-fills the `Add` output, so every tensor has
        // a shape — and the prefix is still not a chain. The shape gate must
        // refuse it (else the phase tile-caps its dense chunks behind a crop
        // every chunk refuses), and the crop must refuse it on its own, even
        // under the fully pinned shapes the pre-gate `prefix_shapes` returned.
        let mut deep = residual;
        deep.ops.truncate(5);
        deep.ops.extend([
            TwinOpSpec::Conv {
                input: 5,
                weight: w(2 * 2 * 9),
                bias: w(2),
                bias_err: vec![0.0; 2],
                weight_rel_err: 1e-15,
                kernel: (2, 2, 3, 3),
                stride: (1, 1),
                pads: (1, 1, 1, 1),
                ishape: (2, 4, 4),
                oshape: (2, 4, 4),
            }, // t6
            TwinOpSpec::Relu { input: 6 },    // t7 (trunk relu 2)
            TwinOpSpec::Flatten { input: 7 }, // t8
            TwinOpSpec::Gemm {
                input: 8,
                weight: w(3 * 32),
                bias: w(3),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (3, 32),
            }, // t9
            TwinOpSpec::Relu { input: 9 },    // t10
            TwinOpSpec::Gemm {
                input: 10,
                weight: w(2 * 3),
                bias: w(2),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 3),
            }, // t11
        ]);
        let net = TwinNet::compile(&deep).expect("compile");
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let relu2 = root.layers[2].op;
        assert!(
            window::prefix_shapes(&net, relu2).is_none(),
            "a residual prefix must be refused at the shape gate even when a downstream conv \
             pins the Add output"
        );
        let pinned: Vec<window::Shape> = vec![(2, 4, 4); relu2 + 2];
        assert!(
            window::prefix_bounds(&net, &root, relu2, &pinned, &[0], None, None).is_none(),
            "the crop must refuse a residual prefix on its own, whatever shapes it is handed"
        );

        let mut rng = StdRng::seed_from_u64(0xB1_3D04);
        let net = TwinNet::compile(&chain_spec(&mut rng, 0.5)).expect("compile");
        let root = RootGates::build(
            &net,
            &vec![-0.3; 128],
            &vec![0.4; 128],
            RoundMode::Outward,
            None,
        )
        .expect("root");
        let relu1 = root.layers[1].op;
        let shapes = window::prefix_shapes(&net, relu1).expect("croppable");
        let n = root.layers[1].n;
        assert!(window::prefix_bounds(&net, &root, relu1, &shapes, &[n + 3], None, None).is_none());
        assert!(window::prefix_bounds(&net, &root, relu1, &shapes, &[], None, None).is_none());
        // A non-relu op is never a seed.
        assert!(window::prefix_bounds(&net, &root, relu1 - 1, &shapes, &[0], None, None).is_none());
    }

    /// DARK BY DEFAULT, and armed-phase containment: the default policy never
    /// sets the crop; an armed phase publishes boxes that contain every sampled
    /// pre-activation and agree with the dense phase up to rounding.
    #[test]
    fn bi_window_phase_is_dark_by_default_and_contained_when_armed() {
        let default_state = BiState::new(
            super::super::BackwardIntermRuntimePolicy {
                enabled: true,
                ..Default::default()
            },
            None,
        );
        assert!(!default_state.window, "#bi-window must ship dark");
        assert_eq!(default_state.win_pairs, 0);

        let mut rng = StdRng::seed_from_u64(0xB1_3D05);
        let net = TwinNet::compile(&chain_spec(&mut rng, 0.6)).expect("compile");
        let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
        let build = |window: bool| {
            RootGates::build_retaining_with_policy(
                &net,
                &lo,
                &hi,
                RoundMode::Outward,
                None,
                None,
                &[],
                super::super::RootPrecisionRuntimePolicy::default(),
                super::super::BackwardIntermRuntimePolicy {
                    enabled: true,
                    secs: 1_000.0,
                    chunk: 16,
                    chunk_secs: 0.0,
                    topk: 4_096,
                    interm_alpha: 2,
                    window,
                    gpu: false,
                    patch: super::super::BiPatchRuntimePolicy::default(),
                },
                None,
            )
            .expect("root")
            .0
        };
        let dense = build(false);
        let cropped = build(true);
        let mut x = Array2::<f64>::zeros((128, 128));
        for i in 0..128 {
            for b in 0..128 {
                let u: f64 = rng.random_range(-1.0..1.0);
                x[[i, b]] = dense.mid[i] + u * dense.rad[i];
            }
        }
        let sel: std::collections::BTreeMap<usize, Vec<usize>> = dense
            .layers
            .iter()
            .map(|layer| (layer.op, (0..layer.n).collect()))
            .collect();
        let (_, pre) = net.forward_points(&x, &sel).expect("forward");
        for (ld, lc) in dense.layers.iter().zip(&cropped.layers) {
            let p = &pre[&ld.op];
            for j in 0..ld.n {
                for b in 0..x.ncols() {
                    let v = p[[j, b]];
                    assert!(
                        v >= lc.l[j] - 1e-9 && v <= lc.u[j] + 1e-9,
                        "armed box misses a sample"
                    );
                }
                let tol = 1e-6 * (1.0 + ld.l[j].abs().max(ld.u[j].abs()));
                assert!(
                    (ld.l[j] - lc.l[j]).abs() <= tol && (ld.u[j] - lc.u[j]).abs() <= tol,
                    "op {} neuron {j}: armed [{}, {}] vs dense [{}, {}] beyond rounding",
                    ld.op,
                    lc.l[j],
                    lc.u[j],
                    ld.l[j],
                    ld.u[j]
                );
            }
        }
    }

    /// ENGAGEMENT, PINNED DIRECTLY. The armed-phase test
    /// above compares the published boxes of a dense and an armed build, and it
    /// would pass IDENTICALLY if the armed phase silently ran every pass dense —
    /// an armed-and-inert crop reads as green there. This drives the phase's
    /// own `tighten_layer` on a chain layer with the crop armed and asserts the
    /// crop's OWN counters: at least one cropped pair ran and no chunk fell
    /// back to dense. What it publishes must still be shrink-only and contain
    /// every sampled pre-activation.
    #[test]
    fn bi_window_phase_enters_the_crop_and_stays_contained() {
        let mut rng = StdRng::seed_from_u64(0xB1_3D06);
        let net = TwinNet::compile(&chain_spec(&mut rng, 0.6)).expect("compile");
        let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let mut x = Array2::<f64>::zeros((128, 64));
        for i in 0..128 {
            for b in 0..64 {
                let u: f64 = rng.random_range(-1.0..1.0);
                x[[i, b]] = root.mid[i] + u * root.rad[i];
            }
        }
        let sel: std::collections::BTreeMap<usize, Vec<usize>> = root
            .layers
            .iter()
            .map(|layer| (layer.op, (0..layer.n).collect()))
            .collect();
        let (_, pre) = net.forward_points(&x, &sel).expect("forward");
        let mut st = BiState::new(
            super::super::BackwardIntermRuntimePolicy {
                enabled: true,
                secs: 1_000.0,
                chunk: 16,
                chunk_secs: 0.0,
                topk: 4_096,
                interm_alpha: 1,
                window: true,
                gpu: false,
                patch: super::super::BiPatchRuntimePolicy::default(),
            },
            None,
        );
        assert!(st.window, "the armed policy must reach the phase state");
        // The first trunk layer past the affine-only layer 0 that has work.
        let li = (1..root.layers.len())
            .find(|&k| {
                let layer = &root.layers[k];
                layer
                    .l
                    .iter()
                    .zip(&layer.u)
                    .any(|(&a, &b)| a < 0.0 && b > 0.0)
            })
            .expect("the fixture has no unstable neuron past layer 0 (VOID)");
        let relu_op = root.layers[li].op;
        let mut prefix: Vec<LayerGates> = root.layers[..li]
            .iter()
            .map(|g| LayerGates {
                op: g.op,
                n: g.n,
                l: g.l.clone(),
                u: g.u.clone(),
                alpha: g.alpha.clone(),
                s: g.s.clone(),
                c: g.c.clone(),
                ms: g.ms.clone(),
                unst: g.unst.clone(),
                clip_rows: None,
            })
            .collect();
        let (l0, u0) = (root.layers[li].l.clone(), root.layers[li].u.clone());
        let (mut l, mut u) = (l0.clone(), u0.clone());
        st.tighten_layer(
            &net,
            RoundMode::Outward,
            &root.mid,
            &root.rad,
            &root.xabs,
            &root.lo,
            &root.hi,
            &mut prefix,
            relu_op,
            &mut l,
            &mut u,
        );
        assert_eq!(prefix.len(), li, "the prefix gates must be handed back");
        assert!(
            st.win_pairs > 0,
            "the armed phase never entered the crop (VOID, not negative)"
        );
        assert_eq!(
            st.win_fallbacks, 0,
            "a single-consumer conv/affine/relu chain must never fall back to dense"
        );
        assert!(st.win_rows_window <= st.win_rows_dense);
        let p = &pre[&relu_op];
        for j in 0..l.len() {
            assert!(
                l[j] >= l0[j] && u[j] <= u0[j],
                "the armed phase widened neuron {j}"
            );
            for b in 0..x.ncols() {
                let v = p[[j, b]];
                assert!(
                    v >= l[j] - 1e-9 && v <= u[j] + 1e-9,
                    "neuron {j}: sampled {v} outside the armed phase's box [{}, {}]",
                    l[j],
                    u[j]
                );
            }
        }
    }

    /// Scheduling helper: every column appears exactly once and every run ends
    /// on a tile boundary.
    #[test]
    fn tile_order_is_a_permutation_with_tile_runs() {
        let cols: Vec<usize> = (0..(3 * 8 * 8)).filter(|j| j % 3 != 1).collect();
        let (ordered, ends) = window::tile_order(&cols, (3, 8, 8), 8);
        let mut sorted = ordered.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, cols, "tile order must be a permutation");
        let mut cursor = 0usize;
        while cursor < ordered.len() {
            let end = ends[cursor];
            assert!(end > cursor && end <= ordered.len());
            assert!(ends[cursor..end].iter().all(|&e| e == end));
            cursor = end;
        }
    }
}
