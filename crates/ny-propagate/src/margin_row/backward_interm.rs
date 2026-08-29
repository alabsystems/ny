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

use ndarray::Array2;
use ny_core::{NyError, Result};

use super::engine::{BackwardEngine, LaneDir, Seed};
use super::net::TwinNet;
use super::root::{LayerGates, RootGates};
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
}

/// Closed execution receipt for one root build's frozen backward-intermediate
/// policy.  Configuration fields identify the treatment; counters prove which
/// values actually constrained live work.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackwardIntermExecutionEvidence {
    pub enabled: bool,
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
    Ok((child, stats))
}

impl BiState {
    pub(crate) fn new(
        policy: super::BackwardIntermRuntimePolicy,
        deadline: Option<Instant>,
    ) -> Self {
        let t0 = Instant::now();
        let cfg = policy.config(RoundMode::Outward).unwrap_or(BiCfg {
            secs: policy.secs,
            chunk: policy.chunk,
            chunk_secs: policy.chunk_secs,
            topk: policy.topk,
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
        }
        Self {
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
        }
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
            root_precision_policy: super::RootPrecisionRuntimePolicy::default(),
            root_precision_execution: super::root::RootPrecisionExecutionEvidence::default(),
        };
        let (mut tightened, mut stabilized, mut crossed, mut done) =
            (0usize, 0usize, 0usize, 0usize);
        let mut seconds_per_col: Option<f64> = None;
        {
            let eng = BackwardEngine::new(net, &tmp);
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
                let mut s = Array2::<f64>::zeros((n_pre, w));
                for (c, &j) in chunk.iter().enumerate() {
                    s[[j, c]] = 1.0;
                }
                // Identity seed is exact: the engine zero-fills its error lane.
                let seed = Seed { s, e: None };
                let passes = (
                    eng.run_prefix(&seed, relu_op, LaneDir::Lower),
                    eng.run_prefix(&seed, relu_op, LaneDir::Upper),
                );
                let (lb, ub) = match passes {
                    (Ok(plo), Ok(pup)) => (eng.concretize_lower(&plo), eng.concretize_upper(&pup)),
                    _ => {
                        // Fail-open: an engine refusal here costs tightness
                        // only. Stop the layer (a failure would repeat).
                        self.work_secs += chunk_t0.elapsed().as_secs_f64();
                        self.pass_errors += 1;
                        break 'chunks;
                    }
                };
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
        self.crossed_total += crossed;
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

    /// Final summary line (engagement telemetry for the whole build).
    pub(crate) fn finish(&self) -> BackwardIntermExecutionEvidence {
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
        BackwardIntermExecutionEvidence {
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
