// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched box-refinement screen for per-clause-input-box disjunctions.
//!
//! nn4sys mscn/lindex band properties are disjunctions of hundreds-to-tens-of-
//! thousands of clauses, each of the form `x ∈ box_k AND Y violates band_k`,
//! where `box_k` is a tiny sub-box (mostly point axes plus one narrow sweep
//! axis). Proving the property UNSAT requires proving EVERY clause impossible
//! over ITS OWN box.
//!
//! The old screen ran one serial IBP + full-CROWN pass per clause and handed
//! the survivors to per-clause BaB, whose shared-root intermediate bounds do
//! not converge under input splitting on these Mul-heavy DAGs — each survivor
//! burned its whole time slice at ~1s per clause, blowing the budget on any
//! instance with more than ~100 clauses.
//!
//! This engine instead:
//! 1. groups clauses that share a (bit-identical) input box, so one bound pass
//!    decides all of that box's rows at once (mscn band pairs `Y <= lo` /
//!    `Y >= hi` share their box);
//! 2. screens waves of boxes in parallel (rayon) with a cheap IBP forward,
//!    falling back to a fresh spec-guided/per-output CROWN pass over the SAME
//!    box for rows IBP cannot close;
//! 3. bisects the widest input axis of any box whose surviving rows are still
//!    open and re-screens the two children — fresh per-box bounds (unlike the
//!    BaB rebound path) converge as the box narrows, so a few levels decide
//!    each band;
//! 4. escalates still-straddling few-axis nodes to the FIRST-ORDER f64
//!    centered form (#f64-mvf, interval forward-mode derivatives — see
//!    `ny-propagate/src/network/graph_ibp_f64_mvf.rs`), whose excess shrinks
//!    QUADRATICALLY with box width: the zeroth-order interval's linear
//!    dependency excess made the mscn `_dual` multi-axis band plateaus cost
//!    ~35k leaves per clause (~10x every official budget, commit 196720ef).
//!
//! SOUNDNESS / row isolation: a clause is marked verified ONLY when every leaf
//! of the partition of its own box proves — from bounds computed over that
//! leaf alone — that one of the clause's own constraints cannot hold there.
//! The two children of a split cover the parent (`[lo, mid]` ∪ `[mid, hi]`),
//! so the leaves cover the clause box. Bounds are never shared across
//! different boxes, and verdicts are never shared across clauses: clauses
//! sharing one bit-identical box share the bound PASS but are each checked
//! against their own constraint rows. Every abort path (deadline, depth cap,
//! node cap, degenerate box, propagation failure) marks the affected clauses
//! UNPROVEN — never verified. This screen can only prove clauses UNSAT; it
//! never produces a SAT/violated verdict.

use ny_core::GemmEngine;
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::Interval64;
use ny_tensor::BoundedTensor;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tracing::{debug, info};

use super::disjunctive_precheck::{
    clause_provably_unsat, constraint_is_interval_checkable, crown_precheck_clauses,
    ibp_output_bounds, tighten_input_to_box,
};
use super::BetaCrownModel;

/// Maximum bisection depth PER WIDE AXIS of a clause box (the effective cap
/// for a box with `k` wide axes is `k * MAX_DEPTH` — each level splits only
/// the widest axis, so a shared flat cap starved multi-axis boxes: an mscn
/// `_dual` clause with 3 sweep axes reached only 8 splits per axis and its
/// f64 enclosure could never close, making the clause structurally unprovable
/// at ANY budget — 41/240 clauses on cardinality_1_240_128_dual).
/// Leaf width per axis = axis width / 2^24 ≈ f32 resolution on unit-scale
/// axes. Bounds converge as boxes narrow, so only the neighborhood of the
/// extremizer stays open — deep lineages are narrow, and the deadline (or the
/// no-deadline node cap) still bounds the total work.
const MAX_DEPTH: u16 = 24;

/// Nodes bounded per parallel wave.
const WAVE_SIZE: usize = 256;

/// #mscn-throughput A/B knobs: env overrides for the wave/chunk sizes so the
/// 7.5x campaign (ledger charter 2026-07-19) can sweep the schedule space
/// without rebuilds. Schedule-only — every setting bounds the same boxes with
/// the same sound walks; unset ⇒ the shipped constants, byte-identical.
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(default)
}

fn wave_size() -> usize {
    env_usize("NY_SCREEN_WAVE_SIZE", WAVE_SIZE)
}

fn cell_chunk() -> usize {
    env_usize("NY_SCREEN_CELL_CHUNK", F64_BATCH_CELL_CHUNK)
}

fn mvf_chunk() -> usize {
    env_usize("NY_SCREEN_MVF_CHUNK", F64_BATCH_MVF_CHUNK)
}

/// #nn4sys-dual gate: certified f64 CROWN tail attempt on near-closing boxes
/// (default OFF — byte-identical when unset).
fn dual_f64_tail_enabled() -> bool {
    std::env::var("NY_DUAL_F64_TAIL").ok().as_deref() == Some("1")
}

/// Smallest f32 at or above `x` — thresholds handed to the f64 tail are
/// compared as `f64::from(t_f32)`, so rounding UP keeps `l > t` at least as
/// strong as the exact `l > x` impossibility test (sound, conservative).
fn f32_at_least(x: f64) -> f32 {
    let c = x as f32;
    if f64::from(c) < x {
        ny_tensor::next_up_f32(c)
    } else {
        c
    }
}

/// Convert ONE clause into the f64 tail's spec form: rows whose certified
/// lower bound exceeding the threshold proves the CONSTRAINT IMPOSSIBLE
/// (mirroring `f64_test_deficit`'s directions):
///   Y_i <= k  impossible iff min Y_i        > k  -> row +e_i,     t = k
///   Y_i >= k  impossible iff min (-Y_i)     > -k -> row -e_i,     t = -k
///   Y_i <= Y_j impossible iff min (Y_i-Y_j) > 0  -> row e_i - e_j, t = 0
///   Y_i >= Y_j impossible iff min (Y_j-Y_i) > 0  -> row e_j - e_i, t = 0
/// (The joint row bound is TIGHTER than the deficit test's decoupled l/u
/// comparison — still sound.) Any non-convertible constraint makes the whole
/// clause skip the tail (`None`): with the tail's SOME-row-per-group verdict,
/// omitting a constraint would be sound but pointless; omitting is only
/// unsound if we dropped a whole clause from a multi-clause call, which the
/// per-clause call structure rules out.
fn clause_to_tail_spec(
    clause: &[OutputConstraint],
    out_dim: usize,
) -> Option<(ndarray::Array2<f32>, Vec<f32>, Vec<usize>)> {
    let mut rows: Vec<f32> = Vec::new();
    let mut ths: Vec<f32> = Vec::new();
    let mut n_rows = 0usize;
    for c in clause {
        let mut row = vec![0.0f32; out_dim];
        match c {
            OutputConstraint::LessEqConst(i, k) | OutputConstraint::LessThanConst(i, k) => {
                if *i >= out_dim {
                    return None;
                }
                row[*i] = 1.0;
                ths.push(f32_at_least(*k));
            }
            OutputConstraint::GreaterEqConst(i, k) | OutputConstraint::GreaterThanConst(i, k) => {
                if *i >= out_dim {
                    return None;
                }
                row[*i] = -1.0;
                ths.push(f32_at_least(-*k));
            }
            OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                if *i >= out_dim || *j >= out_dim || i == j {
                    return None;
                }
                row[*i] = 1.0;
                row[*j] = -1.0;
                ths.push(0.0);
            }
            OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                if *i >= out_dim || *j >= out_dim || i == j {
                    return None;
                }
                row[*i] = -1.0;
                row[*j] = 1.0;
                ths.push(0.0);
            }
            _ => return None,
        }
        rows.extend_from_slice(&row);
        n_rows += 1;
    }
    if n_rows == 0 {
        return None;
    }
    let spec = ndarray::Array2::from_shape_vec((n_rows, out_dim), rows).ok()?;
    Some((spec, ths, vec![n_rows]))
}

/// Boxes per batched zeroth-order chunk (#f64-batch-boxes): bounds one
/// chunk's wall time (the deadline is re-checked between chunks) while
/// keeping the stacked Linear rows (m = chunk·rows) far above the fast
/// Rump kernel's m >= 16 gate. A whole 256-box wave in one chunk: after the
/// fused-lane partition (#f64-fused-walk) this chunk serves only non-MVF
/// boxes, whose per-box cost is a single cell channel, so one chunk stays
/// ~1s of wall between deadline checks while amortizing the per-walk
/// overhead (weight lookups, refcount maps, rayon dispatch) 4x further.
const F64_BATCH_CELL_CHUNK: usize = 256;

/// Boxes per batched fused centered+mono chunk: the fused walk holds a
/// value tensor plus one derivative tensor per seeded axis per live node
/// per box, so chunks stay smaller than the cell lane; 96 (up from the
/// original 32, matching ny-propagate's internal chunk cap) keeps the live
/// set a few hundred MB at the 2048-wide mscn dual shapes while amortizing
/// per-chunk walk overhead 3x further. m = chunk·k·rows still saturates
/// the Rump kernel.
const F64_BATCH_MVF_CHUNK: usize = 96;

/// Deficit (how far a row's sound f64 bound fails its impossibility test)
/// at or below which a still-open row counts as NEAR-CLOSING.
///
/// Two throughput heuristics key off it (soundness is untouched — proofs
/// only ever come from a sound bound excluding a clause's own constraint
/// over the node's own box):
/// - a node ALL of whose open rows are near-closing sits on a band plateau
///   (nn4sys mscn `_dual`: margins ±1e-5, plateau overshoot shrinking
///   linearly with box width) — it splits 4-way instead of binary, skipping
///   one full intermediate front of bound passes per level;
/// - near-closing rows skip the f32 CROWN attempt: their remaining deficit
///   is below the f32 CROWN rounding floor (measured on mscn duals: f32
///   CROWN closes ZERO plateau rows — the pure-f32 screen exhausts its tree
///   at 0/1 clauses), while one or two more cheap f64 levels decide them.
const NEAR_CLOSING_DEFICIT: f64 = 1e-4;

/// Hard cap on total nodes processed (backstop when no deadline is set).
const MAX_NODES: usize = 300_000;

/// Maximum SEEDED input axes (per `ny_propagate::centered_seed_axes_f32` —
/// axes wide enough for a derivative channel; ulp-wide axes from the outward
/// f64→f32 box rounding are absorbed into the center walk instead) for the
/// first-order (mean-value/centered form) f64 bound (#f64-mvf). Each
/// centered pass costs ~(k + 2) cell-forward walks for k seeded axes; the
/// mscn `_dual` clause boxes it exists for have 1-3 sweep axes, so 4 covers
/// them with headroom while keeping the pass cheap. Boxes with more seeded
/// axes use the zeroth-order bound only.
const MVF_MAX_WIDE_AXES: usize = 4;

/// Minimum number of timed in-screen CROWN attempts before the adaptive cost
/// gate may decide to shed CROWN for the rest of the run (see
/// [`CrownCostGate`]).
const CROWN_GATE_SAMPLE: usize = 3;

/// Default per-attempt cost threshold (milliseconds) for the adaptive CROWN
/// gate, chosen between the two MEASURED regimes (per-attempt wall time of
/// in-screen `crown_precheck_clauses` calls inside the rayon wave, VNN-COMP
/// 2025 nn4sys, release build, M-series):
/// - mscn_128d / 128d_dual: 161-169ms (three instances, 240-2260 clauses).
///   CROWN must STAY ENABLED here: it prunes subtrees the f64 lane must
///   otherwise split through — with CROWN cardinality_1_2260_128_dual
///   (100s budget, the tight one) closes in 71.8s / 164,527 nodes; shedding
///   CROWN costs 83.0s / 227,189 nodes, which would blow the ~7s slack
///   measured on the reference machine (~93s).
/// - mscn_2048d / 2048d_dual: 455-512ms on every multi-group instance
///   (247ms on the single-group cardinality_0_1_2048, where shedding is
///   irrelevant — it verifies in ~6s either way). Here CROWN closes ~nothing
///   and DOMINATES the screen: shedding takes cardinality_1_1_2048_dual from
///   127 to 333 nodes in its 20s budget.
///
/// 300ms sits 1.8x above the 128d band and 1.5x below the deep-2048d band.
/// Per-attempt wall time varies with wave contention (same-net attempts are
/// cheaper on low-clause instances), so the margins are real but not huge —
/// NY_SCREEN_CROWN_MS retunes without a rebuild if a new net lands between
/// the bands.
const CROWN_GATE_DEFAULT_MS: f64 = 300.0;

/// Adaptive per-run cost gate for the screen's in-node f32 CROWN attempts.
///
/// CROWN here is a SPEED HEURISTIC only — it selects which sound bound gets
/// attempted on a node, never how a proof is judged — so shedding it is
/// soundness-free: rows it would have closed simply stay open and are decided
/// by cheaper f64 levels or fail conservatively to downstream lanes.
///
/// The gate times each in-screen `crown_precheck_clauses` call; once at least
/// [`CROWN_GATE_SAMPLE`] attempts are recorded, a running average above the
/// threshold sheds CROWN for the REMAINDER of this run. Cheap-CROWN nets
/// (mscn_128d, ~165ms/attempt, where CROWN prunes subtrees and is worth its
/// cost) never trip the default threshold; expensive nets (deep mscn_2048d,
/// ~460-510ms/attempt, where CROWN closes ~nothing and dominated the screen,
/// starving refinement) shed it within the first wave.
///
/// One instance per `refine_clause_boxes` run — no global state. Atomics
/// (Relaxed) because nodes are bounded in parallel waves; the decision is a
/// heuristic, so a lost update or an extra sample is harmless.
///
/// Env `NY_SCREEN_CROWN_MS` tunes the threshold in ms; `0` disables the gate
/// entirely (always allow CROWN); unset/unparseable uses the default.
struct CrownCostGate {
    /// Per-attempt threshold in ms; `None` = gate disabled (always allow).
    threshold_ms: Option<f64>,
    /// Timed attempts required before the shed decision may latch.
    /// [`CROWN_GATE_SAMPLE`] normally; 1 on batch-worthy fat-Linear nets
    /// (#f64-batch-boxes), where every sampled attempt is ~0.5s of pure
    /// cost (2048d mscn: CROWN closes ~nothing, measured) and the wasted
    /// sample was the margin between closing cardinality_1_1_2048_dual
    /// inside its 20s budget and timing out. Still adaptive: a fat-Linear
    /// net whose CROWN is cheap (< threshold) never sheds.
    sample: usize,
    /// While fewer than `sample` attempts have been RECORDED, at most
    /// `sample` attempts may START (batch-worthy fat-Linear nets only): a
    /// first wave otherwise launches a whole rayon front of CROWN attempts
    /// in parallel before the first sample can latch the shed decision —
    /// measured 14 × ~400-476ms of pure thread-time on the mscn_2048d
    /// screens, starving the f64 lane. Skipped nodes simply keep their f64
    /// bounds (CROWN here is a speed heuristic; on these nets it closes
    /// ~nothing, measured) — verdict-safe. Non-batch nets keep the
    /// unlimited legacy behavior byte-for-byte.
    probe_cap: bool,
    /// CROWN attempts STARTED (>= the recorded `attempts`).
    started: AtomicUsize,
    /// Timed CROWN attempts so far this run.
    attempts: AtomicUsize,
    /// Total wall time of those attempts, in nanoseconds.
    total_nanos: AtomicU64,
    /// Latched shed decision: once true, no further CROWN this run.
    shed: AtomicBool,
    /// Nodes whose CROWN attempt was skipped because of the shed decision.
    skips: AtomicUsize,
}

impl CrownCostGate {
    /// Test-only convenience: [`Self::with_threshold_and_sample`] with the
    /// default sample count.
    #[cfg(test)]
    fn with_threshold(threshold_ms: Option<f64>) -> Self {
        Self::with_threshold_sample_probe(threshold_ms, CROWN_GATE_SAMPLE, false)
    }

    fn with_threshold_sample_probe(
        threshold_ms: Option<f64>,
        sample: usize,
        probe_cap: bool,
    ) -> Self {
        Self {
            threshold_ms,
            sample: sample.max(1),
            probe_cap,
            started: AtomicUsize::new(0),
            attempts: AtomicUsize::new(0),
            total_nanos: AtomicU64::new(0),
            shed: AtomicBool::new(false),
            skips: AtomicUsize::new(0),
        }
    }

    /// Batteries-included default-ON with a disable-env: `NY_SCREEN_CROWN_MS`
    /// tunes the threshold, `0` disables the gate. `fast_shed` (batch-worthy
    /// fat-Linear nets) latches the shed decision after ONE over-threshold
    /// attempt instead of [`CROWN_GATE_SAMPLE`] — see the `sample` field —
    /// and caps concurrent attempts at that sample until it is recorded
    /// (`probe_cap`).
    fn from_env(fast_shed: bool) -> Self {
        let threshold_ms = match std::env::var("NY_SCREEN_CROWN_MS") {
            Ok(v) => match v.trim().parse::<f64>() {
                Ok(0.0) => None, // 0 = gate off, always allow
                Ok(ms) if ms.is_finite() && ms > 0.0 => Some(ms),
                _ => Some(CROWN_GATE_DEFAULT_MS),
            },
            Err(_) => Some(CROWN_GATE_DEFAULT_MS),
        };
        Self::with_threshold_sample_probe(
            threshold_ms,
            if fast_shed { 1 } else { CROWN_GATE_SAMPLE },
            fast_shed,
        )
    }

    /// Whether a node that WANTS a CROWN attempt may run one. Counts the
    /// skip when the answer is no (so the run log reports how many attempts
    /// the shed decision saved).
    fn should_attempt(&self) -> bool {
        if self.threshold_ms.is_none() {
            return true;
        }
        if self.shed.load(Ordering::Relaxed) {
            self.skips.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if self.probe_cap && self.attempts.load(Ordering::Relaxed) < self.sample {
            // No decision basis yet: admit only `sample` probe attempts;
            // the rest of the wave skips instead of piling ~0.5s attempts
            // onto every rayon thread (they are counted as skips).
            let started = self.started.fetch_add(1, Ordering::Relaxed);
            if started >= self.sample {
                self.skips.fetch_add(1, Ordering::Relaxed);
                return false;
            }
        }
        true
    }

    /// Record one timed CROWN attempt and evaluate the shed decision.
    fn record(&self, elapsed: Duration) {
        let Some(threshold_ms) = self.threshold_ms else {
            return;
        };
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        let n = self.attempts.fetch_add(1, Ordering::Relaxed) + 1;
        let total = self
            .total_nanos
            .fetch_add(nanos, Ordering::Relaxed)
            .saturating_add(nanos);
        if n >= self.sample {
            let avg_ms = total as f64 / n as f64 / 1e6;
            if avg_ms > threshold_ms && !self.shed.swap(true, Ordering::Relaxed) {
                info!(
                    attempts = n,
                    avg_ms,
                    threshold_ms,
                    "Screen CROWN cost gate: per-attempt cost over threshold, \
                     shedding CROWN for the rest of this run"
                );
            }
        }
    }

    /// (attempts, avg ms over them, shed?, skipped attempts) for the run log.
    fn stats(&self) -> (usize, f64, bool, usize) {
        let attempts = self.attempts.load(Ordering::Relaxed);
        let avg_ms = if attempts == 0 {
            0.0
        } else {
            self.total_nanos.load(Ordering::Relaxed) as f64 / attempts as f64 / 1e6
        };
        (
            attempts,
            avg_ms,
            self.shed.load(Ordering::Relaxed),
            self.skips.load(Ordering::Relaxed),
        )
    }
}

/// One work item: an input box plus the indices of the clauses that still
/// need to be proven impossible over it.
struct BoxNode {
    bounds: BoundedTensor,
    active: Vec<usize>,
    depth: u16,
    /// Per-lineage depth cap: `MAX_DEPTH * (wide axes of the ROOT box)`, so
    /// each wide axis gets its own full bisection budget.
    depth_cap: u16,
}

/// Outcome of bounding one node (computed in parallel, applied serially).
struct NodeOutcome {
    /// Clause indices proven impossible over this node's box.
    proven: Vec<usize>,
    /// Clause indices still open over this node's box.
    unresolved: Vec<usize>,
    /// Children produced by splitting the box (empty when `unresolved` is
    /// empty, the depth cap is reached, or no axis can make progress).
    /// Children always cover the parent box exactly.
    children: Vec<BoundedTensor>,
    /// Depth of the children (parent depth + levels the split advanced:
    /// 1 for a binary split, 2 for a 4-way split).
    child_depth: u16,
    depth_cap: u16,
}

/// Sound f64 bounding context (nn4sys mscn band clauses, #f64-leaf).
///
/// Wraps `GraphNetwork::propagate_ibp_f64_cell` — the sound f64 interval
/// forward (per-op outward widening / Higham dot-product bounds; see
/// `ny-propagate/src/network/graph_ibp_f64_cell.rs`). When available it is
/// the screen's PRIMARY per-node bound (see `bound_node`): the f32
/// forward's outward-rounding floor is WIDER than the true band margins
/// (~1e-5 absolute) of the mscn `_dual` clauses, which no amount of f32
/// refinement can recover, while the f64 floor (~1e-16 relative) decides
/// them — and the f64 pass is also measurably cheaper per node than the
/// f32 IBP on these DAGs.
///
/// Availability is FAIL-CLOSED: only Graph models whose complete
/// output-ancestor op set is supported by the f64 forward
/// (`supports_ibp_f64_cell`), and only unless the `NY_F64_LEAF=0`
/// kill-switch is set. Any f64 propagation error falls back to the f32
/// lane for that node.
struct F64LeafEscalation<'a> {
    graph: &'a ny_propagate::GraphNetwork,
    /// Nodes bounded by the f64 forward.
    attempts: AtomicUsize,
    /// Clause-obligations closed by the f64 enclosure.
    rescued: AtomicUsize,
    /// Whether the FIRST-ORDER centered form (#f64-mvf) is available: full
    /// derivative-rule op support (`supports_ibp_f64_centered`, a strict
    /// subset of the cell op set — discontinuous ops are excluded because the
    /// mean value theorem does not hold across a jump) and no `NY_F64_MVF=0`
    /// kill-switch. Batteries-included default-ON.
    mvf: bool,
    /// Nodes where the centered form was computed.
    mvf_attempts: AtomicUsize,
    /// Clause-obligations closed by the centered form that the zeroth-order
    /// interval could not close on the same node.
    mvf_rescued: AtomicUsize,
    /// Whether the MONOTONICITY-CORNER refinement (#mono-corner) may run:
    /// derivative-sign certification from the SAME interval forward-mode
    /// channels the centered form computes, then sound f64 cell walks over
    /// the sign-pinned corner boxes — EXACT range (up to eval rounding) on
    /// fully-certified boxes, a sound dimension-reduced refinement on
    /// partially-certified ones (see
    /// `ny-propagate/src/network/graph_ibp_f64_mvf.rs` for the enclosure
    /// argument). Requires `mvf`; `NY_MONO_CORNER=0` is the kill-switch
    /// (batteries-included default-ON).
    mono: bool,
    /// Nodes where the mono-corner walk ran (fused with the centered walk).
    mono_attempts: AtomicUsize,
    /// Clause-obligations closed by the mono bound that neither the
    /// zeroth-order interval nor the centered form could close.
    mono_rescued: AtomicUsize,
    /// (output element, seeded axis) pairs whose derivative sign certified,
    /// and the total pairs examined — the run's measured certification rate.
    mono_cert_pairs: AtomicUsize,
    mono_total_pairs: AtomicUsize,
    /// Nodes where EVERY (element, axis) pair certified (the mono bound is
    /// the exact range there).
    mono_all_certified: AtomicUsize,
    /// Whether the batched multi-box f64 forward (#f64-batch-boxes) may be
    /// used for whole waves: stacks a wave's boxes into fat interval GEMMs
    /// so the Rump kernel fires at mscn's thin per-box Linear shapes. ON
    /// only for graphs with a fat Linear (`f64_batch_worthwhile` — measured
    /// win at 2048-wide mscn, measured LOSS at 128-wide where per-box rayon
    /// parallelism beats serial batched GEMM passes); batteries-included
    /// default-ON there, `NY_F64_BATCH_BOXES=0` disables.
    batch: bool,
    /// Prepared f64 Linear weights (exact `Wᵀ` + `|Wᵀ|` per node), built
    /// lazily at the first batched wave and reused for the whole run — the
    /// per-call weight conversion/split otherwise dominates thin-m batched
    /// GEMMs. Purely a speed cache: kernel results are bit-identical.
    weight_cache: OnceLock<ny_propagate::F64WeightCache>,
    /// Waves bounded through the batched multi-box forward.
    batched_waves: AtomicUsize,
    /// Boxes bounded through the batched multi-box forward.
    batched_boxes: AtomicUsize,
    /// Waves where the batched forward failed and every box fell back to the
    /// per-box walk (byte-identical legacy behavior).
    batched_fallbacks: AtomicUsize,
    /// Boxes whose centered form was computed by the batched multi-box
    /// centered pass (#f64-batch-boxes × #f64-mvf).
    batched_mvf_boxes: AtomicUsize,
    /// Waves where the batched centered pass failed and MVF candidates fell
    /// back to per-node centered walks (byte-identical legacy behavior).
    batched_mvf_fallbacks: AtomicUsize,
}

/// Build the escalation context if the model and environment allow it.
fn f64_leaf_escalation(model_net: &BetaCrownModel) -> Option<F64LeafEscalation<'_>> {
    // Batteries-included default-ON; NY_F64_LEAF=0 is the kill-switch.
    if std::env::var("NY_F64_LEAF").is_ok_and(|v| v == "0") {
        return None;
    }
    match model_net {
        BetaCrownModel::Graph(graph) if graph.supports_ibp_f64_cell() => {
            let mvf = graph.supports_ibp_f64_centered()
                && !std::env::var("NY_F64_MVF").is_ok_and(|v| v == "0");
            let mono = mvf && !std::env::var("NY_MONO_CORNER").is_ok_and(|v| v == "0");
            Some(F64LeafEscalation {
                graph,
                attempts: AtomicUsize::new(0),
                rescued: AtomicUsize::new(0),
                mvf,
                mvf_attempts: AtomicUsize::new(0),
                mvf_rescued: AtomicUsize::new(0),
                mono,
                mono_attempts: AtomicUsize::new(0),
                mono_rescued: AtomicUsize::new(0),
                mono_cert_pairs: AtomicUsize::new(0),
                mono_total_pairs: AtomicUsize::new(0),
                mono_all_certified: AtomicUsize::new(0),
                batch: ny_propagate::batch_boxes_enabled() && graph.f64_batch_worthwhile(),
                weight_cache: OnceLock::new(),
                batched_waves: AtomicUsize::new(0),
                batched_boxes: AtomicUsize::new(0),
                batched_fallbacks: AtomicUsize::new(0),
                batched_mvf_boxes: AtomicUsize::new(0),
                batched_mvf_fallbacks: AtomicUsize::new(0),
            })
        }
        _ => None,
    }
}

/// Record one mono-corner walk's certification stats (#mono-corner).
fn record_mono_stats(esc: &F64LeafEscalation<'_>, out: &ny_propagate::CenteredMono) {
    esc.mono_attempts.fetch_add(1, Ordering::Relaxed);
    esc.mono_cert_pairs
        .fetch_add(out.certified_pairs, Ordering::Relaxed);
    esc.mono_total_pairs
        .fetch_add(out.total_pairs, Ordering::Relaxed);
    if out.all_certified {
        esc.mono_all_certified.fetch_add(1, Ordering::Relaxed);
    }
}

/// Count of input axes the centered form would seed (delegates to the
/// seeding rule in ny-propagate so gate and walk agree), or `None` when the
/// tensor is not contiguous (skip the centered form for it).
fn wide_axis_count(bounds: &BoundedTensor) -> Option<usize> {
    let lo = bounds.lower();
    let hi = bounds.upper();
    let lo_s = lo.as_slice()?;
    let hi_s = hi.as_slice()?;
    Some(ny_propagate::centered_seed_axes_f32(lo_s, hi_s))
}

/// Elementwise intersection of two SOUND output enclosures — sound, and
/// nonempty whenever both really enclose the (nonempty) true image. Returns
/// `None` (fail closed: caller keeps the zeroth-order interval) if any
/// element intersects empty, which would indicate an enclosure bug.
fn intersect_intervals(a: &Interval64, b: &Interval64) -> Option<Interval64> {
    if a.lower.shape() != b.lower.shape() {
        return None;
    }
    let mut lower = a.lower.clone();
    let mut upper = a.upper.clone();
    let mut ok = true;
    ndarray::Zip::from(&mut lower)
        .and(&mut upper)
        .and(&b.lower)
        .and(&b.upper)
        .for_each(|l, u, &bl, &bu| {
            let nl = l.max(bl);
            let nu = u.min(bu);
            // NaN-preserving fail-closed gate: a NaN endpoint must reject the
            // intersection (`nl > nu` would accept it).
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(nl <= nu) {
                ok = false;
            } else {
                *l = nl;
                *u = nu;
            }
        });
    ok.then_some(Interval64 { lower, upper })
}

/// Whether sound f64 output bounds prove a single (unsafe-region) constraint
/// can NEVER hold over the leaf box. Mirrors the f32
/// `disjunctive_precheck::constraint_provably_false`, but both the bound and
/// the spec constant are f64 so the comparison is exact at f64 precision.
/// NaN bounds (conservative saturation upstream) fail every comparison —
/// fail-closed by construction.
fn constraint_provably_false_f64(output: &Interval64, c: &OutputConstraint) -> bool {
    let l = |i: usize| output.lower.iter().nth(i).copied();
    let u = |i: usize| output.upper.iter().nth(i).copied();
    match c {
        OutputConstraint::LessEqConst(i, k) => l(*i).is_some_and(|v| v > *k),
        OutputConstraint::LessThanConst(i, k) => l(*i).is_some_and(|v| v >= *k),
        OutputConstraint::GreaterEqConst(i, k) => u(*i).is_some_and(|v| v < *k),
        OutputConstraint::GreaterThanConst(i, k) => u(*i).is_some_and(|v| v <= *k),
        OutputConstraint::LessEq(i, j) => matches!((l(*i), u(*j)), (Some(a), Some(b)) if a > b),
        OutputConstraint::LessThan(i, j) => matches!((l(*i), u(*j)), (Some(a), Some(b)) if a >= b),
        OutputConstraint::GreaterEq(i, j) => matches!((u(*i), l(*j)), (Some(a), Some(b)) if a < b),
        OutputConstraint::GreaterThan(i, j) => {
            matches!((u(*i), l(*j)), (Some(a), Some(b)) if a <= b)
        }
        _ => false,
    }
}

/// A clause is provably impossible over the leaf iff ANY of its constraints
/// is provably false there (same conjunction rule as the f32 screen).
fn clause_provably_unsat_f64(output: &Interval64, clause: &[OutputConstraint]) -> bool {
    !clause.is_empty()
        && clause
            .iter()
            .any(|c| constraint_provably_false_f64(output, c))
}

/// How far a constraint's f64 impossibility test FAILED (distance from the
/// decisive f64 bound to the threshold), or `None` when the test passed /
/// the bound is unavailable / the variant is not interval-checkable.
fn f64_test_deficit(output: &Interval64, c: &OutputConstraint) -> Option<f64> {
    let l = |i: usize| output.lower.iter().nth(i).copied();
    let u = |i: usize| output.upper.iter().nth(i).copied();
    let d = match c {
        // Impossible iff l_i > k — deficit is how far l_i sits at/below k.
        OutputConstraint::LessEqConst(i, k) | OutputConstraint::LessThanConst(i, k) => {
            l(*i).map(|v| *k - v)
        }
        // Impossible iff u_i < k — deficit is how far u_i sits at/above k.
        OutputConstraint::GreaterEqConst(i, k) | OutputConstraint::GreaterThanConst(i, k) => {
            u(*i).map(|v| v - *k)
        }
        // Impossible iff l_i > u_j.
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => match (l(*i), u(*j)) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        },
        // Impossible iff u_i < l_j.
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            match (u(*i), l(*j)) {
                (Some(a), Some(b)) => Some(a - b),
                _ => None,
            }
        }
        _ => None,
    };
    d.filter(|v| *v >= 0.0)
}

/// Whether a still-open clause is NEAR-CLOSING under the f64 bound: some
/// constraint of its own missed its impossibility test by no more than
/// `NEAR_CLOSING_DEFICIT`. See the constant for how this steers splitting
/// and the CROWN attempt.
fn clause_near_closing_f64(output: &Interval64, clause: &[OutputConstraint]) -> bool {
    clause
        .iter()
        .any(|c| f64_test_deficit(output, c).is_some_and(|d| d <= NEAR_CLOSING_DEFICIT))
}

/// Midpoint plausibility gate for the CROWN attempt under f64-primary
/// bounding: true when the f64 interval's CENTER already satisfies some
/// impossibility test of the clause — i.e. the center estimate says
/// "impossible", only the interval width blocks the proof. Mirrors the f32
/// `crown_plausibly_closes`.
fn crown_plausibly_closes_f64(output: &Interval64, clause: &[OutputConstraint]) -> bool {
    let mid = |i: usize| -> Option<f64> {
        let l = output.lower.iter().nth(i).copied()?;
        let u = output.upper.iter().nth(i).copied()?;
        Some(l + (u - l) * 0.5)
    };
    clause.iter().any(|c| match c {
        OutputConstraint::LessEqConst(i, k) | OutputConstraint::LessThanConst(i, k) => {
            mid(*i).is_some_and(|m| m > *k)
        }
        OutputConstraint::GreaterEqConst(i, k) | OutputConstraint::GreaterThanConst(i, k) => {
            mid(*i).is_some_and(|m| m < *k)
        }
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
            matches!((mid(*i), mid(*j)), (Some(a), Some(b)) if a > b)
        }
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            matches!((mid(*i), mid(*j)), (Some(a), Some(b)) if a < b)
        }
        _ => false,
    })
}

/// Bit-exact key for grouping clauses that share an input box.
fn box_key(bounds: &BoundedTensor) -> Option<Vec<u32>> {
    let lo = bounds.lower().as_slice()?;
    let hi = bounds.upper().as_slice()?;
    let mut key = Vec::with_capacity(lo.len() * 2);
    key.extend(lo.iter().map(|v| v.to_bits()));
    key.extend(hi.iter().map(|v| v.to_bits()));
    Some(key)
}

/// Bisect the widest axis of `bounds` at its midpoint. Returns `None` when no
/// axis has positive finite width or the f32 midpoint cannot make strict
/// progress (children cover the parent: `[lo, mid]` ∪ `[mid, hi]`).
fn split_widest_axis(bounds: &BoundedTensor) -> Option<(BoundedTensor, BoundedTensor)> {
    let lo = bounds.lower();
    let hi = bounds.upper();
    let lo_s = lo.as_slice()?;
    let hi_s = hi.as_slice()?;

    let mut best_idx = None;
    let mut best_width = 0.0f32;
    for (i, (&l, &h)) in lo_s.iter().zip(hi_s.iter()).enumerate() {
        let w = h - l;
        if w.is_finite() && w > best_width {
            best_width = w;
            best_idx = Some(i);
        }
    }
    let idx = best_idx?;
    let (l, h) = (lo_s[idx], hi_s[idx]);
    let mid = l + (h - l) * 0.5;
    if !(mid > l && mid < h) {
        // Width at f32 resolution — cannot make progress.
        return None;
    }

    let mut left_hi = hi.clone();
    let mut right_lo = lo.clone();
    *left_hi.as_slice_mut()?.get_mut(idx)? = mid;
    *right_lo.as_slice_mut()?.get_mut(idx)? = mid;
    let left = BoundedTensor::new(lo.clone(), left_hi).ok()?;
    let right = BoundedTensor::new(right_lo, hi.clone()).ok()?;
    Some((left, right))
}

/// Split the widest axis of `bounds` into up to 4 segments at the quarter
/// points. Children cover the parent exactly (`[l,c1] ∪ [c1,c2] ∪ … ∪ [cK,h]`
/// for whatever strictly-interior cuts survive f32 rounding). Used for
/// f64-only flat-zone descent, where every intermediate binary level costs a
/// full front of bound passes: a 4-way split advances two levels at once,
/// skipping the intermediate front. Returns `None` when no cut can make
/// strict progress (defer to `split_widest_axis`'s `None` semantics).
fn split_widest_axis_4way(bounds: &BoundedTensor) -> Option<Vec<BoundedTensor>> {
    let lo = bounds.lower();
    let hi = bounds.upper();
    let lo_s = lo.as_slice()?;
    let hi_s = hi.as_slice()?;

    let mut best_idx = None;
    let mut best_width = 0.0f32;
    for (i, (&l, &h)) in lo_s.iter().zip(hi_s.iter()).enumerate() {
        let w = h - l;
        if w.is_finite() && w > best_width {
            best_width = w;
            best_idx = Some(i);
        }
    }
    let idx = best_idx?;
    let (l, h) = (lo_s[idx], hi_s[idx]);
    let w = h - l;
    let mut cuts: Vec<f32> = Vec::with_capacity(3);
    for q in [0.25f32, 0.5, 0.75] {
        let c = l + w * q;
        if c > l && c < h && cuts.last().is_none_or(|&prev| c > prev) {
            cuts.push(c);
        }
    }
    if cuts.is_empty() {
        return None;
    }

    let mut children = Vec::with_capacity(cuts.len() + 1);
    let mut seg_lo = l;
    for &cut in &cuts {
        let mut child_lo = lo.clone();
        let mut child_hi = hi.clone();
        *child_lo.as_slice_mut()?.get_mut(idx)? = seg_lo;
        *child_hi.as_slice_mut()?.get_mut(idx)? = cut;
        children.push(BoundedTensor::new(child_lo, child_hi).ok()?);
        seg_lo = cut;
    }
    let mut last_lo = lo.clone();
    *last_lo.as_slice_mut()?.get_mut(idx)? = seg_lo;
    children.push(BoundedTensor::new(last_lo, hi.clone()).ok()?);
    Some(children)
}

/// Whether a fresh CROWN pass over this box plausibly closes some clause that
/// IBP could not: true when the IBP interval's MIDPOINT already satisfies the
/// impossibility test for one of the clause's constraints — i.e. IBP's center
/// estimate says "impossible", only the interval width blocks the proof.
/// High in the tree (center estimate inside the unsafe region) CROWN is
/// hopeless and this gate skips its cost — on mscn_2048d a per-node CROWN is
/// ~30ms vs ~3ms IBP, and spending it on wide boxes starved the refinement.
fn crown_plausibly_closes(ibp: &BoundedTensor, clause: &[OutputConstraint]) -> bool {
    let lo = ibp.lower();
    let hi = ibp.upper();
    let mid = |i: usize| -> Option<f32> {
        let l = lo.iter().nth(i).copied()?;
        let u = hi.iter().nth(i).copied()?;
        Some(l + (u - l) * 0.5)
    };
    clause.iter().any(|c| match c {
        OutputConstraint::LessEqConst(i, k) | OutputConstraint::LessThanConst(i, k) => {
            mid(*i).is_some_and(|m| f64::from(m) > *k)
        }
        OutputConstraint::GreaterEqConst(i, k) | OutputConstraint::GreaterThanConst(i, k) => {
            mid(*i).is_some_and(|m| f64::from(m) < *k)
        }
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
            matches!((mid(*i), mid(*j)), (Some(a), Some(b)) if a > b)
        }
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            matches!((mid(*i), mid(*j)), (Some(a), Some(b)) if a < b)
        }
        _ => false,
    })
}

/// Bound one node.
///
/// f64-PRIMARY: when the sound f64 forward is available (`F64LeafEscalation`
/// built — Graph model, full op support, no kill-switch), it IS the per-node
/// bound: it is strictly tighter than the f32 IBP (rounding floor ~1e-16 vs
/// ~3e-5 on the mscn graphs — the f32 floor sits ABOVE the ±1e-5 `_dual`
/// band margins, making f32 refinement structurally dead near band
/// extremizers) and measured ~10x cheaper per pass on the nn4sys mscn DAGs.
/// The f32 IBP+CROWN lane remains for models without f64 support and, fail-
/// closed, for any node where the f64 propagation errors.
#[allow(clippy::too_many_arguments)]
fn bound_node(
    node: &BoxNode,
    model_net: &BetaCrownModel,
    clauses: &[Vec<OutputConstraint>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    f64_leaf: Option<&F64LeafEscalation<'_>>,
    crown_gate: &CrownCostGate,
    precomputed_f64: Option<&Interval64>,
    precomputed_centered: Option<&Interval64>,
    precomputed_mono: Option<&Interval64>,
) -> NodeOutcome {
    // Deadline short-circuit: past the deadline this node's still-open
    // clauses are lost either way (the wave loop breaks next iteration and
    // fails every queued obligation), so don't start expensive per-node
    // walks — return the conservative "unresolved, no children" outcome,
    // which marks this node's clauses unproven exactly like the deadline
    // tail does. Bounds the wave's overshoot past the official timeout to
    // the work already in flight (a 256-node wave of per-node MVF fallbacks
    // was measured to overshoot a 40s budget by ~4s).
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return NodeOutcome {
            proven: Vec::new(),
            unresolved: node.active.clone(),
            children: Vec::new(),
            child_depth: node.depth,
            depth_cap: node.depth_cap,
        };
    }
    if let Some(esc) = f64_leaf {
        if let Some(outcome) = bound_node_f64(
            node,
            esc,
            model_net,
            clauses,
            gemm_engine,
            deadline,
            crown_gate,
            precomputed_f64,
            precomputed_centered,
            precomputed_mono,
        ) {
            return outcome;
        }
    }
    bound_node_f32(node, model_net, clauses, gemm_engine, deadline, crown_gate)
}

/// f64-primary bounding: one sound f64 interval pass decides this node's
/// rows; a gated f32 CROWN attempt covers wide boxes where the linear
/// relaxation can out-prune interval bounds. Returns `None` if the f64
/// propagation fails (caller falls back to the f32 lane for this node).
///
/// `precomputed` carries this node's zeroth-order f64 output when the wave
/// was bounded through the batched multi-box forward (#f64-batch-boxes) —
/// sound and per-box-isolated by that walk's contract; `None` computes the
/// identical per-box walk here. `precomputed_centered`/`precomputed_mono`
/// carry the batched fused walk's first-order and mono-corner bounds the
/// same way.
#[allow(clippy::too_many_arguments)]
fn bound_node_f64(
    node: &BoxNode,
    esc: &F64LeafEscalation<'_>,
    model_net: &BetaCrownModel,
    clauses: &[Vec<OutputConstraint>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    crown_gate: &CrownCostGate,
    precomputed: Option<&Interval64>,
    precomputed_centered: Option<&Interval64>,
    precomputed_mono: Option<&Interval64>,
) -> Option<NodeOutcome> {
    esc.attempts.fetch_add(1, Ordering::Relaxed);
    let input64 = Interval64::from_f32(node.bounds.lower(), node.bounds.upper());
    // Fused zeroth+centered walk (#f64-fused-walk): when this node's MVF
    // escalation is going to be consulted anyway (per-box lane, seeded-axis
    // gate — it fires on ~99% of mscn nodes), ONE mean-value walk yields BOTH
    // the zeroth-order bound (its value channel is bit-identical to
    // `propagate_ibp_f64_cell`; ny-propagate gate
    // `centered_with_value_matches_separate_walks`) and the centered bound,
    // instead of a cell walk plus a centered walk that re-derives the
    // identical value channel internally (2+k walks per node instead of
    // 3+k). Clause decisions are unchanged: `out64` and the centered
    // interval are bit-identical to the unfused path's.
    let mut fused_centered: Option<Interval64> = None;
    let mut fused_mono: Option<Interval64> = None;
    let mut out64 = match precomputed {
        Some(out) => out.clone(),
        None => {
            let mvf_wanted = esc.mvf
                && precomputed_centered.is_none()
                && !node.active.is_empty()
                && wide_axis_count(&node.bounds)
                    .is_some_and(|w| (1..=MVF_MAX_WIDE_AXES).contains(&w));
            // Fused walk, with the mono-corner refinement (#mono-corner)
            // riding on the same derivative channels when enabled: `value`
            // and `centered` are bit-identical either way (ny-propagate
            // contract), so clause decisions below are unchanged unless the
            // strictly-tighter mono bound closes MORE rows.
            let fused = if mvf_wanted && esc.mono {
                match esc.graph.propagate_ibp_f64_centered_mono(&input64) {
                    Ok(out) => {
                        record_mono_stats(esc, &out);
                        fused_mono = out.mono;
                        Some((out.value, out.centered))
                    }
                    Err(_) => None,
                }
            } else if mvf_wanted {
                esc.graph
                    .propagate_ibp_f64_centered_with_value(&input64)
                    .ok()
            } else {
                None
            };
            match fused {
                Some((value, centered)) => {
                    fused_centered = Some(centered);
                    value
                }
                // No escalation wanted for this node — or the centered walk
                // failed (a per-node retry would deterministically fail the
                // same way): the plain zeroth-order walk, exactly as before.
                None => esc.graph.propagate_ibp_f64_cell(&input64).ok()?,
            }
        }
    };

    let mut proven = Vec::new();
    let mut unresolved = Vec::new();
    for &c in &node.active {
        if clause_provably_unsat_f64(&out64, &clauses[c]) {
            proven.push(c);
        } else {
            unresolved.push(c);
        }
    }
    esc.rescued.fetch_add(proven.len(), Ordering::Relaxed);

    // FIRST-ORDER escalation (#f64-mvf): rows still open here have a
    // zeroth-order interval straddling their clause threshold. On few-axis
    // boxes the mean-value/centered form `f64(mid) ± Σ_i |D_i|·r_i`
    // (interval forward-mode derivatives; see
    // `ny-propagate/src/network/graph_ibp_f64_mvf.rs` for the enclosure
    // proof) shrinks QUADRATICALLY with the box instead of linearly — the
    // measured blocker for the mscn `_dual` multi-axis plateau clauses
    // (~35k zeroth-order leaves each, ~10x every official budget). The
    // centered pass already intersects with the zeroth-order interval
    // internally; the defensive re-intersection here keeps this node's
    // bound sound even if either enclosure were buggy, and any failure
    // keeps the zeroth-order bound (fail-closed).
    if esc.mvf && !unresolved.is_empty() {
        let wide = wide_axis_count(&node.bounds);
        if wide.is_some_and(|w| (1..=MVF_MAX_WIDE_AXES).contains(&w)) {
            esc.mvf_attempts.fetch_add(1, Ordering::Relaxed);
            // Batched-wave centered result when available (#f64-batch-boxes,
            // sound + per-box-isolated by that walk's contract); otherwise
            // the fused walk's centered bound (#f64-fused-walk, bit-identical
            // to the per-node centered walk). The per-node walk itself runs
            // only when the zeroth bound was precomputed by the batched lane
            // (so no fused walk happened) — the byte-identical legacy
            // fallback; when the FUSED walk failed, a retry would
            // deterministically fail the same way, so keep the zeroth bound.
            let centered = match precomputed_centered {
                Some(c) => Some(c.clone()),
                None => match fused_centered.take() {
                    Some(c) => Some(c),
                    None if precomputed.is_some() => {
                        esc.graph.propagate_ibp_f64_centered(&input64).ok()
                    }
                    None => None,
                },
            };
            if let Some(centered) = centered {
                if let Some(tightened) = intersect_intervals(&out64, &centered) {
                    out64 = tightened;
                    let mut still_open = Vec::with_capacity(unresolved.len());
                    let mut closed = 0usize;
                    for &c in &unresolved {
                        if clause_provably_unsat_f64(&out64, &clauses[c]) {
                            proven.push(c);
                            closed += 1;
                        } else {
                            still_open.push(c);
                        }
                    }
                    unresolved = still_open;
                    esc.mvf_rescued.fetch_add(closed, Ordering::Relaxed);
                }
            }
        }
    }

    // MONOTONICITY-CORNER escalation (#mono-corner): rows still open after
    // the centered form get the corner bound — EXACT (up to sound eval
    // rounding) on boxes whose derivative signs fully certify, which the
    // lindex learned-index bands are by construction. The bound rides the
    // FUSED per-box walk above, or arrives precomputed from the batched
    // fused walk (#f64-batch-boxes: pattern corner walks stacked across the
    // chunk — the per-node fallback fused walk stays deliberately absent on
    // batched-lane nodes, whose W·(k+2) THIN walks were measured to cost
    // cardinality_1_1_2048_dual its whole 20s budget).
    if esc.mono && !unresolved.is_empty() {
        let mono_iv = match precomputed_mono {
            Some(m) => Some(m.clone()),
            None => fused_mono.take(),
        };
        if let Some(m) = mono_iv {
            if let Some(tightened) = intersect_intervals(&out64, &m) {
                out64 = tightened;
                let mut still_open = Vec::with_capacity(unresolved.len());
                let mut closed = 0usize;
                for &c in &unresolved {
                    if clause_provably_unsat_f64(&out64, &clauses[c]) {
                        proven.push(c);
                        closed += 1;
                    } else {
                        still_open.push(c);
                    }
                }
                unresolved = still_open;
                esc.mono_rescued.fetch_add(closed, Ordering::Relaxed);
            }
        }
    }

    // f32 CROWN attempt for rows the f64 interval could not close, on boxes
    // where it plausibly helps: midpoint already on the impossible side
    // (only interval width blocks the proof) AND the row is NOT near-closing
    // (a near-closing row's remaining deficit is below the f32 CROWN
    // rounding floor — measured on the mscn duals, f32 CROWN closes zero
    // plateau rows — and one or two more cheap f64 levels decide it).
    let crown_rows: Vec<usize> = unresolved
        .iter()
        .copied()
        .filter(|&c| {
            crown_plausibly_closes_f64(&out64, &clauses[c])
                && !clause_near_closing_f64(&out64, &clauses[c])
        })
        .collect();
    if !crown_rows.is_empty()
        && deadline.is_none_or(|d| Instant::now() < d)
        && crown_gate.should_attempt()
    {
        let survivor_clauses: Vec<Vec<OutputConstraint>> =
            crown_rows.iter().map(|&c| clauses[c].clone()).collect();
        let crown_start = Instant::now();
        let crown_verified = crown_precheck_clauses(
            model_net,
            &node.bounds,
            &survivor_clauses,
            &[],
            gemm_engine,
            deadline,
        );
        crown_gate.record(crown_start.elapsed());
        let closed: Vec<usize> = crown_rows
            .iter()
            .zip(crown_verified.iter())
            .filter_map(|(&c, &v)| v.then_some(c))
            .collect();
        if !closed.is_empty() {
            unresolved.retain(|c| {
                if closed.contains(c) {
                    proven.push(*c);
                    false
                } else {
                    true
                }
            });
        }
    }

    // #nn4sys-dual f64 tail attempt (dark, NY_DUAL_F64_TAIL=1): near-closing
    // rows are exactly the ones the f32 CROWN attempt skips (their deficit is
    // below the f32 rounding floor). Give each ONE certified f64 CROWN shot
    // (graph_crown_f64_tail, with the Sigmoid/Div substitution arms) before
    // bisecting. Sound: the tail's `true` is a full certified-outward f64
    // refutation of every constraint row passed; `false` changes nothing.
    if !unresolved.is_empty()
        && dual_f64_tail_enabled()
        && deadline.is_none_or(|d| Instant::now() < d)
    {
        if let BetaCrownModel::Graph(graph) = model_net {
            let out_dim = out64.lower.len();
            let tail_closed: Vec<usize> = unresolved
                .iter()
                .copied()
                .filter(|&c| clause_near_closing_f64(&out64, &clauses[c]))
                .filter(|&c| {
                    clause_to_tail_spec(&clauses[c], out_dim).is_some_and(|(spec, ths, sizes)| {
                        ny_propagate::network::f64_tail_box_attempt(
                            graph,
                            &node.bounds,
                            &spec,
                            &ths,
                            &sizes,
                            gemm_engine,
                            deadline,
                        )
                    })
                })
                .collect();
            if !tail_closed.is_empty() {
                unresolved.retain(|c| {
                    if tail_closed.contains(c) {
                        proven.push(*c);
                        false
                    } else {
                        true
                    }
                });
            }
        }
    }

    // Band-plateau nodes (every open row near-closing) advance TWO levels
    // per split (4-way), skipping one intermediate front of bound passes.
    let in_flat_zone = !unresolved.is_empty()
        && unresolved
            .iter()
            .all(|&c| clause_near_closing_f64(&out64, &clauses[c]));
    let (children, child_depth) = split_children(node, in_flat_zone, &unresolved);

    Some(NodeOutcome {
        proven,
        unresolved,
        children,
        child_depth,
        depth_cap: node.depth_cap,
    })
}

/// Legacy f32 lane: IBP over the box, then (for rows IBP cannot close, when
/// plausibly useful) a fresh CROWN pass over the SAME box via the shared
/// precheck entry — which stacks the surviving rows of this box as rows of
/// ONE spec-guided pass when they are relational, or checks them against
/// per-output bounds otherwise. Used when the f64 forward is unavailable
/// (non-Graph model, unsupported op, `NY_F64_LEAF=0`) or errored for this
/// node — in which regime there is nothing to escalate leaves to, so rows an
/// unsplittable/depth-capped leaf cannot close simply stay unproven.
fn bound_node_f32(
    node: &BoxNode,
    model_net: &BetaCrownModel,
    clauses: &[Vec<OutputConstraint>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    crown_gate: &CrownCostGate,
) -> NodeOutcome {
    let mut proven = Vec::new();
    let mut unresolved = Vec::new();

    let ibp = ibp_output_bounds(model_net, &node.bounds, gemm_engine);
    for &c in &node.active {
        match &ibp {
            Some(out) if clause_provably_unsat(out, &clauses[c]) => proven.push(c),
            _ => unresolved.push(c),
        }
    }

    // Fresh CROWN attempt over this box for the rows IBP could not close.
    // (Tighter than IBP; recomputes intermediates for THIS box, so it keeps
    // converging under bisection.) Gated on the IBP midpoint test so wide
    // boxes split instead of paying for a hopeless CROWN; when IBP itself
    // failed there is no midpoint to consult and CROWN is the only bound.
    let crown_worthwhile = match &ibp {
        Some(out) => unresolved
            .iter()
            .any(|&c| crown_plausibly_closes(out, &clauses[c])),
        None => true,
    };
    if !unresolved.is_empty()
        && crown_worthwhile
        && deadline.is_none_or(|d| Instant::now() < d)
        && crown_gate.should_attempt()
    {
        let survivor_clauses: Vec<Vec<OutputConstraint>> =
            unresolved.iter().map(|&c| clauses[c].clone()).collect();
        let crown_start = Instant::now();
        let crown_verified = crown_precheck_clauses(
            model_net,
            &node.bounds,
            &survivor_clauses,
            &[],
            gemm_engine,
            deadline,
        );
        crown_gate.record(crown_start.elapsed());
        let mut still_open = Vec::with_capacity(unresolved.len());
        for (&c, verified) in unresolved.iter().zip(crown_verified.iter()) {
            if *verified {
                proven.push(c);
            } else {
                still_open.push(c);
            }
        }
        unresolved = still_open;
    }

    let (children, child_depth) = split_children(node, false, &unresolved);

    NodeOutcome {
        proven,
        unresolved,
        children,
        child_depth,
        depth_cap: node.depth_cap,
    }
}

/// Produce children for a node's still-open rows: binary bisection of the
/// widest axis, or a 4-way split (advancing two levels at once) on band
/// plateaus. Empty when nothing is open, the depth cap is reached, or no
/// axis can make strict progress.
fn split_children(
    node: &BoxNode,
    four_way: bool,
    unresolved: &[usize],
) -> (Vec<BoundedTensor>, u16) {
    if unresolved.is_empty() || node.depth >= node.depth_cap {
        return (Vec::new(), node.depth);
    }
    if four_way {
        match split_widest_axis_4way(&node.bounds) {
            Some(kids) => {
                let step = if kids.len() > 2 { 2 } else { 1 };
                (kids, node.depth.saturating_add(step))
            }
            None => (Vec::new(), node.depth),
        }
    } else {
        match split_widest_axis(&node.bounds) {
            Some((l, r)) => (vec![l, r], node.depth.saturating_add(1)),
            None => (Vec::new(), node.depth),
        }
    }
}

/// Batched box-refinement screen. Returns one flag per clause: `true` means
/// the clause is PROVEN unsatisfiable over its own (tightened) input box.
/// Every failure/abort path yields `false` (unproven) for the affected
/// clauses — this function never over-claims.
pub(super) fn refine_clause_boxes(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Vec<bool> {
    refine_clause_boxes_counted(
        model_net,
        input,
        clauses,
        per_clause_input_bounds,
        gemm_engine,
        deadline,
    )
    .0
}

/// [`refine_clause_boxes`] plus the number of box nodes processed — the
/// budget-shape assertions of the plateau-clause tests key on it.
fn refine_clause_boxes_counted(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> (Vec<bool>, usize) {
    let start = Instant::now();
    let n = clauses.len();
    // Sound f64 leaf escalation (Graph models with a fully-supported op set
    // only; NY_F64_LEAF=0 disables). None => leaves fail exactly as before.
    let f64_leaf = f64_leaf_escalation(model_net);
    // Adaptive per-run CROWN cost gate (NY_SCREEN_CROWN_MS tunes; 0 = off).
    let crown_gate = CrownCostGate::from_env(f64_leaf.as_ref().is_some_and(|esc| esc.batch));
    // open_nodes[c]: number of live boxes whose proof obligation still covers
    // clause c. failed[c]: some part of c's box could not be proven (any
    // abort/cap path). Verified at the end ⇔ all obligations discharged and
    // none failed.
    let mut open_nodes = vec![0usize; n];
    let mut failed = vec![false; n];

    // Group clauses by bit-identical tightened box so band pairs share one
    // bound pass per box. Clauses with no interval-checkable constraint can
    // never be proven by this screen — fail them up front instead of
    // refining their box for nothing.
    let mut groups: HashMap<Vec<u32>, (BoundedTensor, Vec<usize>)> = HashMap::new();
    for (idx, clause) in clauses.iter().enumerate() {
        if !clause.iter().any(constraint_is_interval_checkable) {
            failed[idx] = true;
            continue;
        }
        let tightened = match per_clause_input_bounds.get(idx) {
            Some(b) if !b.is_empty() => tighten_input_to_box(input, b),
            _ => input.clone(),
        };
        match box_key(&tightened) {
            Some(key) => {
                open_nodes[idx] += 1;
                groups
                    .entry(key)
                    .or_insert_with(|| (tightened, Vec::new()))
                    .1
                    .push(idx);
            }
            None => failed[idx] = true, // non-contiguous tensor — cannot screen
        }
    }

    let num_groups = groups.len();
    let mut queue: Vec<BoxNode> = groups
        .into_values()
        .map(|(bounds, active)| {
            // Depth budget scales with the ROOT box's wide axes: every level
            // bisects only the widest axis, so `k` wide axes share the
            // lineage depth — a flat cap left multi-axis mscn `_dual` boxes
            // at 2^-8-wide axes, unprovable at any budget.
            let wide_axes = match (bounds.lower().as_slice(), bounds.upper().as_slice()) {
                (Some(lo), Some(hi)) => lo
                    .iter()
                    .zip(hi.iter())
                    .filter(|(&l, &h)| h > l)
                    .count()
                    .max(1) as u16,
                _ => 1,
            };
            BoxNode {
                bounds,
                active,
                depth: 0,
                depth_cap: MAX_DEPTH.saturating_mul(wide_axes),
            }
        })
        .collect();

    let mut nodes_processed = 0usize;
    let mut deadline_hit = false;

    while !queue.is_empty() {
        // MAX_NODES is the no-deadline backstop only. When a deadline is set
        // it is the sole budget: the mscn `_dual` instances with 100s+
        // timeouts legitimately need >300k nodes (the 240-clause instance
        // alone hit the cap at 14/240 clauses with 20s of budget left).
        if deadline.is_some_and(|d| Instant::now() >= d)
            || (deadline.is_none() && nodes_processed >= MAX_NODES)
        {
            deadline_hit = true;
            break;
        }

        let wave_len = queue.len().min(wave_size());
        let mut wave: Vec<BoxNode> = queue.drain(queue.len() - wave_len..).collect();
        nodes_processed += wave.len();

        // Drop clauses that already failed elsewhere: their global proof is
        // already lost (failed[c] pins the final flag to false regardless of
        // open_nodes bookkeeping), so refining for them is wasted budget.
        for node in &mut wave {
            node.active.retain(|&c| !failed[c]);
        }

        // Batched multi-box f64 pass (#f64-batch-boxes): bound the wave's
        // boxes in chunked DAG walks that stack them into fat interval GEMMs
        // (the Rump kernel fires at m = boxes*rows where the per-box thin
        // shapes stay on the scalar loop), against the run's prepared f64
        // weights. Sound and per-box-isolated by the walks' contracts; ANY
        // chunk failure (or the deadline) leaves those boxes on the
        // byte-identical per-box lane.
        //
        // Fused-lane partition (#f64-fused-walk): boxes the per-box lane
        // would bound with the fused zeroth+centered walk anyway (live
        // clauses + a seedable-axis count inside the MVF gate) go straight
        // to the batched FUSED walk — value channel included, bit-identical
        // to the batched zeroth walk (ny-propagate gate
        // `batched_fused_value_matches_batched_cells`), centered and
        // mono-corner bounds riding the same derivative channels
        // (#mono-corner: pattern corner walks stacked across the chunk) —
        // instead of a zeroth chunk plus a centered chunk that re-derives
        // the identical value channel. Measured (cardinality_0_500_2048,
        // 15s probe): 92% of batched boxes took the centered pass, so the
        // separate zeroth pass was almost pure recomputation.
        let mut cell_outs: Vec<Option<Interval64>> = vec![None; wave.len()];
        let mut centered_outs: Vec<Option<Interval64>> = vec![None; wave.len()];
        let mut mono_outs: Vec<Option<Interval64>> = vec![None; wave.len()];
        if let Some(esc) = f64_leaf.as_ref().filter(|e| e.batch && wave.len() >= 2) {
            let weights = esc
                .weight_cache
                .get_or_init(|| esc.graph.build_f64_weight_cache());
            esc.batched_waves.fetch_add(1, Ordering::Relaxed);
            let (mvf_idx, cell_idx): (Vec<usize>, Vec<usize>) = if esc.mvf {
                (0..wave.len()).partition(|&i| {
                    !wave[i].active.is_empty()
                        && wide_axis_count(&wave[i].bounds)
                            .is_some_and(|w| (1..=MVF_MAX_WIDE_AXES).contains(&w))
                })
            } else {
                (Vec::new(), (0..wave.len()).collect())
            };
            for chunk in cell_idx.chunks(cell_chunk()) {
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    break;
                }
                let inputs: Vec<Interval64> = chunk
                    .iter()
                    .map(|&i| Interval64::from_f32(wave[i].bounds.lower(), wave[i].bounds.upper()))
                    .collect();
                match esc
                    .graph
                    .propagate_ibp_f64_cells_cached(&inputs, Some(weights))
                {
                    Ok(outs) => {
                        esc.batched_boxes.fetch_add(outs.len(), Ordering::Relaxed);
                        for (&i, out) in chunk.iter().zip(outs) {
                            cell_outs[i] = Some(out);
                        }
                    }
                    Err(_) => {
                        esc.batched_fallbacks.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            for chunk in mvf_idx.chunks(mvf_chunk()) {
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    break;
                }
                let boxes: Vec<Interval64> = chunk
                    .iter()
                    .map(|&i| Interval64::from_f32(wave[i].bounds.lower(), wave[i].bounds.upper()))
                    .collect();
                match esc.graph.propagate_ibp_f64_centered_mono_cells_cached(
                    &boxes,
                    esc.mono,
                    Some(weights),
                ) {
                    Ok(outs) => {
                        esc.batched_boxes.fetch_add(outs.len(), Ordering::Relaxed);
                        esc.batched_mvf_boxes
                            .fetch_add(outs.len(), Ordering::Relaxed);
                        for (&i, out) in chunk.iter().zip(outs) {
                            if esc.mono {
                                record_mono_stats(esc, &out);
                            }
                            cell_outs[i] = Some(out.value);
                            centered_outs[i] = Some(out.centered);
                            mono_outs[i] = out.mono;
                        }
                    }
                    Err(_) => {
                        // Whole chunk falls back to the per-node lane
                        // (bound_node's fused per-box walk — equivalent
                        // bounds, byte-identical decisions).
                        esc.batched_mvf_fallbacks.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        let outcomes: Vec<NodeOutcome> = wave
            .par_iter()
            .enumerate()
            .map(|(i, node)| {
                bound_node(
                    node,
                    model_net,
                    clauses,
                    gemm_engine,
                    deadline,
                    f64_leaf.as_ref(),
                    &crown_gate,
                    cell_outs[i].as_ref(),
                    centered_outs[i].as_ref(),
                    mono_outs[i].as_ref(),
                )
            })
            .collect();

        for outcome in outcomes {
            for &c in &outcome.proven {
                open_nodes[c] -= 1;
            }
            let open = &outcome.unresolved[..];
            if open.is_empty() {
                continue;
            }
            if outcome.children.is_empty() {
                // Depth cap or unsplittable box: these clauses cannot be
                // fully proven — conservative unproven.
                for &c in open {
                    failed[c] = true;
                    open_nodes[c] -= 1;
                }
            } else {
                // Each open clause's obligation moves from the parent to
                // ALL K children: net +(K-1) open nodes per clause.
                for &c in open {
                    open_nodes[c] += outcome.children.len() - 1;
                }
                for child in outcome.children {
                    queue.push(BoxNode {
                        bounds: child,
                        active: open.to_vec(),
                        depth: outcome.child_depth,
                        depth_cap: outcome.depth_cap,
                    });
                }
            }
        }
    }

    if deadline_hit {
        for node in &queue {
            for &c in &node.active {
                failed[c] = true;
            }
        }
    }

    let result: Vec<bool> = (0..n).map(|c| !failed[c] && open_nodes[c] == 0).collect();
    let verified = result.iter().filter(|&&v| v).count();
    let (f64_attempts, f64_rescued, mvf_available, mvf_attempts, mvf_rescued) = f64_leaf
        .as_ref()
        .map(|esc| {
            (
                esc.attempts.load(Ordering::Relaxed),
                esc.rescued.load(Ordering::Relaxed),
                esc.mvf,
                esc.mvf_attempts.load(Ordering::Relaxed),
                esc.mvf_rescued.load(Ordering::Relaxed),
            )
        })
        .unwrap_or((0, 0, false, 0, 0));
    let (batch_enabled, batched_waves, batched_boxes, batched_fallbacks, mvf_boxes, mvf_falls) =
        f64_leaf
            .as_ref()
            .map(|esc| {
                (
                    esc.batch,
                    esc.batched_waves.load(Ordering::Relaxed),
                    esc.batched_boxes.load(Ordering::Relaxed),
                    esc.batched_fallbacks.load(Ordering::Relaxed),
                    esc.batched_mvf_boxes.load(Ordering::Relaxed),
                    esc.batched_mvf_fallbacks.load(Ordering::Relaxed),
                )
            })
            .unwrap_or((false, 0, 0, 0, 0, 0));
    let (mono_enabled, mono_attempts, mono_rescued, mono_cert, mono_total, mono_all) = f64_leaf
        .as_ref()
        .map(|esc| {
            (
                esc.mono,
                esc.mono_attempts.load(Ordering::Relaxed),
                esc.mono_rescued.load(Ordering::Relaxed),
                esc.mono_cert_pairs.load(Ordering::Relaxed),
                esc.mono_total_pairs.load(Ordering::Relaxed),
                esc.mono_all_certified.load(Ordering::Relaxed),
            )
        })
        .unwrap_or((false, 0, 0, 0, 0, 0));
    let (crown_attempts, crown_avg_ms, crown_shed, crown_skips) = crown_gate.stats();
    info!(
        clauses = n,
        verified,
        groups = num_groups,
        nodes = nodes_processed,
        deadline_hit,
        f64_leaf_available = f64_leaf.is_some(),
        f64_leaf_attempts = f64_attempts,
        f64_leaf_rescued = f64_rescued,
        f64_mvf_available = mvf_available,
        f64_mvf_attempts = mvf_attempts,
        f64_mvf_rescued = mvf_rescued,
        mono_corner_enabled = mono_enabled,
        mono_corner_attempts = mono_attempts,
        mono_corner_rescued = mono_rescued,
        mono_corner_cert_pairs = mono_cert,
        mono_corner_total_pairs = mono_total,
        mono_corner_all_certified_nodes = mono_all,
        f64_batch_enabled = batch_enabled,
        f64_batched_waves = batched_waves,
        f64_batched_boxes = batched_boxes,
        f64_batched_fallbacks = batched_fallbacks,
        f64_batched_mvf_boxes = mvf_boxes,
        f64_batched_mvf_fallbacks = mvf_falls,
        crown_gate_enabled = crown_gate.threshold_ms.is_some(),
        crown_attempts,
        crown_avg_ms,
        crown_shed,
        crown_skips,
        elapsed_s = start.elapsed().as_secs_f64(),
        "Box-refinement clause screen complete"
    );
    if verified < n {
        debug!(
            unproven = n - verified,
            "Box-refinement screen left clauses unproven (falling to downstream lanes)"
        );
    }
    (result, nodes_processed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    /// Serializes tests that mutate process-global env kill-switches
    /// (NY_F64_LEAF / NY_F64_MVF) so parallel test threads cannot poison each
    /// other's escalation availability.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn boxed(lo: Vec<f32>, hi: Vec<f32>) -> BoundedTensor {
        let shape = vec![lo.len()];
        BoundedTensor::new(
            ArrayD::from_shape_vec(shape.clone(), lo).unwrap(),
            ArrayD::from_shape_vec(shape, hi).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn split_widest_axis_bisects_the_ranged_axis_and_covers_parent() {
        let b = boxed(vec![0.0, 1.0, -2.0], vec![0.0, 1.0, 2.0]);
        let (l, r) = split_widest_axis(&b).expect("splittable");
        let l_lo = l.lower().as_slice().unwrap().to_vec();
        let l_hi = l.upper().as_slice().unwrap().to_vec();
        let r_lo = r.lower().as_slice().unwrap().to_vec();
        let r_hi = r.upper().as_slice().unwrap().to_vec();
        // Point axes untouched.
        assert_eq!(l_lo[0], 0.0);
        assert_eq!(l_hi[0], 0.0);
        assert_eq!(r_lo[1], 1.0);
        assert_eq!(r_hi[1], 1.0);
        // Ranged axis: children cover the parent and share the midpoint.
        assert_eq!(l_lo[2], -2.0);
        assert_eq!(r_hi[2], 2.0);
        assert_eq!(l_hi[2], r_lo[2]);
        assert!(l_hi[2] > -2.0 && l_hi[2] < 2.0);
    }

    #[test]
    fn split_widest_axis_refuses_point_boxes() {
        let b = boxed(vec![0.5, 1.0], vec![0.5, 1.0]);
        assert!(split_widest_axis(&b).is_none());
    }

    #[test]
    fn box_key_groups_identical_boxes_only() {
        let a = boxed(vec![0.0, 1.0], vec![0.5, 1.0]);
        let b = boxed(vec![0.0, 1.0], vec![0.5, 1.0]);
        let c = boxed(vec![0.0, 1.0], vec![0.6, 1.0]);
        assert_eq!(box_key(&a), box_key(&b));
        assert_ne!(box_key(&a), box_key(&c));
    }

    /// y = (1/3)·x + 0.7 as a 1-node Graph model. 1/3 and 0.7 have no exact
    /// f32 representation, so the sound f32 forward carries a rounding floor
    /// of ~1 f32 ULP (~6e-8 at |y|≈0.8) — wider than the 1e-9 band margins
    /// these tests use.
    fn build_band_graph() -> ny_propagate::GraphNetwork {
        let w = ndarray::arr2(&[[1.0f32 / 3.0]]);
        let b = ndarray::arr1(&[0.7f32]);
        let linear = ny_propagate::layers::LinearLayer::new(w, Some(b)).unwrap();
        let mut graph = ny_propagate::GraphNetwork::new();
        graph.add_node(ny_propagate::GraphNode::from_input(
            "lin",
            ny_propagate::Layer::Linear(linear),
        ));
        graph.set_output("lin");
        graph
    }

    /// Task test (c): a synthetic band clause whose margin (1e-9) sits BELOW
    /// the f32 rounding floor — f32 IBP cannot prove it impossible, the sound
    /// f64 graph IBP can.
    #[test]
    fn f64_leaf_proves_band_clause_below_f32_floor() {
        let graph = build_band_graph();
        let x = 0.3f32;
        let point = boxed(vec![x], vec![x]);

        // True value (f32 weights widen to f64 exactly; one f64 mul + add
        // introduce <= 1e-16 relative error — negligible vs the 1e-9 margin).
        let y = f64::from(1.0f32 / 3.0) * f64::from(x) + f64::from(0.7f32);
        let k = y + 1e-9;
        let clause = vec![OutputConstraint::GreaterEqConst(0, k)];

        // Sound f32 IBP: its outward rounding floor straddles k — unprovable.
        let model = BetaCrownModel::Graph(Box::new(graph.clone()));
        let out32 = ibp_output_bounds(&model, &point, None).expect("f32 IBP");
        assert!(
            !clause_provably_unsat(&out32, &clause),
            "f32 floor should straddle the 1e-9 band margin (upper {} vs k {k})",
            out32.upper().iter().next().unwrap()
        );

        // Sound f64 IBP: rounding floor ~1e-16 — proves the clause impossible.
        let point64 = Interval64::from_f32(point.lower(), point.upper());
        let out64 = graph.propagate_ibp_f64_cell(&point64).expect("f64 IBP");
        assert!(
            clause_provably_unsat_f64(&out64, &clause),
            "f64 enclosure should prove Y >= k impossible (upper {} vs k {k})",
            out64.upper[[0]]
        );
        // And the f64 enclosure still CONTAINS the true value (soundness).
        assert!(out64.lower[[0]] <= y && y <= out64.upper[[0]]);
    }

    /// End-to-end: the box refinement escalates unsplittable (f32-resolution)
    /// leaves to the f64 graph IBP and proves the band clause; with the
    /// NY_F64_LEAF=0 kill-switch the same clause stays unproven.
    #[test]
    fn refine_escalates_f32_resolution_leaves_to_f64() {
        let _env = ENV_LOCK.lock().unwrap();
        let graph = build_band_graph();
        let model = BetaCrownModel::Graph(Box::new(graph));
        let input = boxed(vec![0.0], vec![1.0]);

        // Clause box: a few f32 ULPs around 0.3 (exactly representable
        // endpoints, so tighten_input_to_box is exact).
        let lb32 = 0.3f32;
        let mut ub32 = lb32;
        for _ in 0..4 {
            ub32 = ny_tensor::next_up_f32(ub32);
        }
        let mut clause_box = std::collections::BTreeMap::new();
        clause_box.insert(0usize, (f64::from(lb32), f64::from(ub32)));

        // Band threshold: 1e-9 above the true maximum over the box — below
        // the f32 floor, far above the f64 floor.
        let y_max = f64::from(1.0f32 / 3.0) * f64::from(ub32) + f64::from(0.7f32);
        let k = y_max + 1e-9;
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, k)]];
        let boxes = vec![clause_box];

        let verified = refine_clause_boxes(&model, &input, &clauses, &boxes, None, None);
        assert_eq!(
            verified,
            vec![true],
            "f64 leaf escalation should close the sub-f32-floor band clause"
        );

        // Kill-switch: NY_F64_LEAF=0 must fail-closed back to unproven.
        // (Serialized + restored via the blessed env choke point.)
        let verified_off =
            ny_test_utils::env::with_serialized_env_vars(&[("NY_F64_LEAF", "0")], || {
                refine_clause_boxes(&model, &input, &clauses, &boxes, None, None)
            });
        assert_eq!(
            verified_off,
            vec![false],
            "with the kill-switch the f32-floor clause must stay unproven"
        );
    }

    /// 2-axis band-PLATEAU DAG over input [2] on the box [0.5, 1.5]^2:
    ///
    ///   f(x0, x1) = [relu(x0 + x1) - (x0 + x1)] + 0.25 * sigmoid(x0 - x1)
    ///
    /// The bracket is IDENTICALLY ZERO on the box (x0 + x1 >= 1 > 0) but the
    /// zeroth-order interval cannot see the cancellation: its dependency
    /// excess is ~2*(w0 + w1) — proving a band with margin 1e-9 would need
    /// leaf axes ~5e-10 wide, below both the f32 split resolution and the
    /// depth cap: structurally unprovable, exactly like the measured mscn
    /// `_dual` multi-axis plateau clauses (35k+ leaves, commit 196720ef).
    /// The centered form sees D_i = (1 - 1) + 0.25*sigma' with only
    /// ulp/hull-level slack on the plateau part, so its excess is quadratic
    /// in the box width and the clause closes in a small tree.
    fn build_plateau_graph() -> ny_propagate::GraphNetwork {
        use ny_propagate::layers::{
            AddLayer, MulConstantLayer, ReLULayer, SigmoidLayer, SliceLayer, SubLayer,
        };
        let mut g = ny_propagate::GraphNetwork::new();
        g.add_node(ny_propagate::GraphNode::from_input(
            "x0",
            ny_propagate::Layer::Slice(SliceLayer::new(0, 0, 1)),
        ));
        g.add_node(ny_propagate::GraphNode::from_input(
            "x1",
            ny_propagate::Layer::Slice(SliceLayer::new(0, 1, 2)),
        ));
        g.add_node(ny_propagate::GraphNode::binary(
            "s",
            ny_propagate::Layer::Add(AddLayer),
            "x0",
            "x1",
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "r",
            ny_propagate::Layer::ReLU(ReLULayer),
            vec!["s".to_string()],
        ));
        g.add_node(ny_propagate::GraphNode::binary(
            "plateau",
            ny_propagate::Layer::Sub(SubLayer),
            "r",
            "s",
        ));
        g.add_node(ny_propagate::GraphNode::binary(
            "diff",
            ny_propagate::Layer::Sub(SubLayer),
            "x0",
            "x1",
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "sig",
            ny_propagate::Layer::Sigmoid(SigmoidLayer::new()),
            vec!["diff".to_string()],
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "scaled",
            ny_propagate::Layer::MulConstant(MulConstantLayer::new(ArrayD::from_elem(
                ndarray::IxDyn(&[1]),
                0.25f32,
            ))),
            vec!["sig".to_string()],
        ));
        g.add_node(ny_propagate::GraphNode::binary(
            "out",
            ny_propagate::Layer::Add(AddLayer),
            "plateau",
            "scaled",
        ));
        g.set_output("out");
        g
    }

    /// The plateau clause: `Y >= sup f + 1e-9` over [0.5, 1.5]^2 —
    /// sup f = 0.25*sigma(1) at the (1.5, 0.5) corner; the 1e-9 band margin
    /// sits below every f32 floor (interval or CROWN) but far above the f64
    /// centered form's ulp-level plateau slack.
    fn plateau_clause_and_box() -> (
        Vec<Vec<OutputConstraint>>,
        Vec<std::collections::BTreeMap<usize, (f64, f64)>>,
    ) {
        let sup = 0.25 * (1.0f64 / (1.0 + (-1.0f64).exp()));
        let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, sup + 1e-9)]];
        let mut clause_box = std::collections::BTreeMap::new();
        clause_box.insert(0usize, (0.5f64, 1.5f64));
        clause_box.insert(1usize, (0.5f64, 1.5f64));
        (clauses, vec![clause_box])
    }

    /// Task test (c): an isolated 2-axis plateau clause — structurally
    /// unprovable for the zeroth-order screen (the mscn `_dual` blocker
    /// needed ~35k leaves per clause even when provable) — closes under the
    /// first-order centered form in a SMALL tree (target < 2k nodes).
    #[test]
    fn refine_mvf_closes_two_axis_plateau_clause_cheaply() {
        let _env = ENV_LOCK.lock().unwrap();
        let graph = build_plateau_graph();
        assert!(graph.supports_ibp_f64_centered());
        let model = BetaCrownModel::Graph(Box::new(graph));
        let input = boxed(vec![0.0, 0.0], vec![2.0, 2.0]);
        let (clauses, boxes) = plateau_clause_and_box();

        let (verified, nodes) =
            refine_clause_boxes_counted(&model, &input, &clauses, &boxes, None, None);
        assert_eq!(
            verified,
            vec![true],
            "centered form should prove the 2-axis plateau band clause"
        );
        assert!(
            nodes < 2_000,
            "plateau clause should close in far fewer nodes than the ~35k \
             zeroth-order blow-up (got {nodes})"
        );
    }

    /// Kill-switch contrast: with NY_F64_MVF=0 the same plateau clause is
    /// structurally unprovable (zeroth-order dependency excess ~2*(w0+w1)
    /// never reaches the 1e-9 margin before the depth cap / f32 split
    /// resolution) — the screen must leave it unproven, never mis-verify.
    #[test]
    fn refine_mvf_kill_switch_leaves_plateau_unproven() {
        let _env = ENV_LOCK.lock().unwrap();
        let graph = build_plateau_graph();
        let model = BetaCrownModel::Graph(Box::new(graph));
        let input = boxed(vec![0.0, 0.0], vec![2.0, 2.0]);
        let (clauses, boxes) = plateau_clause_and_box();

        // (Serialized + restored via the blessed env choke point.)
        let verified = ny_test_utils::env::with_serialized_env_vars(&[("NY_F64_MVF", "0")], || {
            let deadline = Some(Instant::now() + Duration::from_secs(5));
            refine_clause_boxes(&model, &input, &clauses, &boxes, None, deadline)
        });
        assert_eq!(
            verified,
            vec![false],
            "without the centered form the sub-f32-floor plateau clause must stay unproven"
        );
    }

    /// y = x·x − x as a 1-input Graph model: monotone NON-DECREASING on
    /// [1, 2] (y' = 2x − 1 ∈ [1, 3], sign-certifiable by the derivative
    /// channels), true range [0, 2], with real curvature — the zeroth-order
    /// interval is [−1, 3] and the centered form [−0.75, 2.25], both far
    /// from the true min at ANY root-box width.
    fn build_mono_quadratic_graph() -> ny_propagate::GraphNetwork {
        use ny_propagate::NETWORK_INPUT;
        let mut g = ny_propagate::GraphNetwork::new();
        g.add_node(ny_propagate::GraphNode::new(
            "prod",
            ny_propagate::Layer::MulBinary(ny_propagate::layers::MulBinaryLayer),
            vec![NETWORK_INPUT.to_string(), NETWORK_INPUT.to_string()],
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "out",
            ny_propagate::Layer::Sub(ny_propagate::layers::SubLayer),
            vec!["prod".to_string(), NETWORK_INPUT.to_string()],
        ));
        g.set_output("out");
        g
    }

    /// #mono-corner: a clause whose impossibility threshold (−1e-9) sits
    /// just below the true minimum 0 — provable ONLY by a near-exact lower
    /// bound. The centered form's excess at the root is ~0.75 and shrinks
    /// quadratically, needing a deep split tree; the f32 CROWN floor
    /// (~1e-7) also misses it. The mono-corner bound is exact up to f64
    /// eval rounding (~1e-16) and must close the clause AT THE ROOT — one
    /// node, zero splits.
    #[test]
    fn refine_mono_corner_closes_monotone_curved_clause_at_root() {
        let _env = ENV_LOCK.lock().unwrap();
        let graph = build_mono_quadratic_graph();
        assert!(graph.supports_ibp_f64_centered());
        let model = BetaCrownModel::Graph(Box::new(graph));
        let input = boxed(vec![1.0], vec![2.0]);
        let clauses = vec![vec![OutputConstraint::LessEqConst(0, -1e-9)]];
        let mut clause_box = std::collections::BTreeMap::new();
        clause_box.insert(0usize, (1.0f64, 2.0f64));
        let boxes = vec![clause_box];

        let (verified, nodes) =
            refine_clause_boxes_counted(&model, &input, &clauses, &boxes, None, None);
        assert_eq!(
            verified,
            vec![true],
            "mono-corner exact bound must prove the sub-centered-margin clause"
        );
        assert_eq!(
            nodes, 1,
            "the corner bound is exact at the ROOT box — no split may be needed"
        );

        // Kill-switch contrast: without the mono bound the screen needs a
        // real split tree (quadratic centered convergence to a 1e-9 margin)
        // — strictly more nodes, same verdict.
        // (Serialized + restored via the blessed env choke point.)
        let (verified_off, nodes_off) =
            ny_test_utils::env::with_serialized_env_vars(&[("NY_MONO_CORNER", "0")], || {
                refine_clause_boxes_counted(&model, &input, &clauses, &boxes, None, None)
            });
        assert_eq!(verified_off, vec![true], "centered+splitting still sound");
        assert!(
            nodes_off > nodes,
            "without mono-corner the same clause must cost splits (got {nodes_off})"
        );
    }

    /// Expensive per-attempt CROWN (mscn_2048d regime, ~100ms >> 20ms) must
    /// shed after the sample, and the skip counter must track the saved
    /// attempts; the sample itself must always be allowed.
    #[test]
    fn crown_gate_sheds_expensive_crown_after_sample() {
        let gate = CrownCostGate::with_threshold(Some(20.0));
        for _ in 0..CROWN_GATE_SAMPLE {
            assert!(gate.should_attempt(), "sampling attempts must be allowed");
            gate.record(Duration::from_millis(100));
        }
        assert!(!gate.should_attempt(), "gate should shed after 3x100ms");
        assert!(!gate.should_attempt());
        let (attempts, avg_ms, shed, skips) = gate.stats();
        assert_eq!(attempts, CROWN_GATE_SAMPLE);
        assert!(avg_ms > 20.0);
        assert!(shed);
        assert_eq!(skips, 2);
    }

    /// Fast-shed gate (batch-worthy fat-Linear nets, production
    /// configuration: sample = 1 AND the probe cap): one over-threshold
    /// attempt latches the shed; a cheap attempt never does.
    #[test]
    fn crown_gate_fast_shed_sheds_after_one_expensive_attempt() {
        let gate = CrownCostGate::with_threshold_sample_probe(Some(20.0), 1, true);
        assert!(gate.should_attempt());
        gate.record(Duration::from_millis(100));
        assert!(!gate.should_attempt(), "fast-shed must latch after 1x100ms");

        let cheap = CrownCostGate::with_threshold_sample_probe(Some(20.0), 1, true);
        cheap.record(Duration::from_millis(1));
        assert!(cheap.should_attempt(), "cheap CROWN must stay enabled");
    }

    /// Probe-cap window semantics (batch-worthy fat-Linear nets): while the
    /// first probe attempt is in flight, the rest of the wave SKIPS (counted
    /// as skips); once the probe records under threshold the gate reopens.
    /// A disabled gate (threshold None) bypasses the cap entirely — the
    /// NY_SCREEN_CROWN_MS=0 kill-switch must never inherit the cap.
    #[test]
    fn crown_gate_probe_cap_limits_first_wave_then_reopens() {
        let gate = CrownCostGate::with_threshold_sample_probe(Some(20.0), 1, true);
        assert!(gate.should_attempt(), "the probe attempt must be admitted");
        assert!(
            !gate.should_attempt(),
            "wave-mates must skip while the probe is unrecorded"
        );
        assert!(!gate.should_attempt());
        let (_, _, _, skips) = gate.stats();
        assert_eq!(skips, 2, "probe-window skips must be counted");
        gate.record(Duration::from_millis(1)); // cheap probe: no shed
        assert!(
            gate.should_attempt(),
            "gate must reopen after the probe records"
        );
        let (_, _, shed, _) = gate.stats();
        assert!(!shed);

        let disabled = CrownCostGate::with_threshold_sample_probe(None, 1, true);
        for _ in 0..4 {
            assert!(
                disabled.should_attempt(),
                "disabled gate must ignore the probe cap (kill-switch contract)"
            );
        }
    }

    /// Cheap per-attempt CROWN (mscn_128d regime, ~1ms << 20ms) must keep
    /// CROWN enabled for the whole run.
    #[test]
    fn crown_gate_keeps_cheap_crown() {
        let gate = CrownCostGate::with_threshold(Some(20.0));
        for _ in 0..50 {
            assert!(gate.should_attempt());
            gate.record(Duration::from_millis(1));
        }
        let (attempts, _, shed, skips) = gate.stats();
        assert_eq!(attempts, 50);
        assert!(!shed, "1ms attempts must never trip a 20ms gate");
        assert_eq!(skips, 0);
    }

    /// A disabled gate (NY_SCREEN_CROWN_MS=0 => threshold None) always
    /// allows CROWN regardless of measured cost.
    #[test]
    fn crown_gate_disabled_always_allows() {
        let gate = CrownCostGate::with_threshold(None);
        for _ in 0..CROWN_GATE_SAMPLE + 2 {
            assert!(gate.should_attempt());
            gate.record(Duration::from_secs(1));
        }
        let (attempts, _, shed, skips) = gate.stats();
        assert_eq!(attempts, 0, "disabled gate records nothing");
        assert!(!shed);
        assert_eq!(skips, 0);
    }

    /// NY_SCREEN_CROWN_MS env contract: unset => default, number => that
    /// threshold, 0 => disabled, garbage => default (never panics).
    #[test]
    fn crown_gate_env_parsing() {
        let _env = ENV_LOCK.lock().unwrap();
        // Serialized + restored via the blessed env choke point (clippy env
        // wall).
        ny_test_utils::env::with_env_edits(|env| {
            env.remove("NY_SCREEN_CROWN_MS");
            assert_eq!(
                CrownCostGate::from_env(false).threshold_ms,
                Some(CROWN_GATE_DEFAULT_MS)
            );
            env.set("NY_SCREEN_CROWN_MS", "5.5");
            assert_eq!(CrownCostGate::from_env(false).threshold_ms, Some(5.5));
            env.set("NY_SCREEN_CROWN_MS", "0");
            assert_eq!(CrownCostGate::from_env(false).threshold_ms, None);
            env.set("NY_SCREEN_CROWN_MS", "not-a-number");
            assert_eq!(
                CrownCostGate::from_env(false).threshold_ms,
                Some(CROWN_GATE_DEFAULT_MS)
            );
        });
    }

    /// The schedule-only A/B knobs accept trimmed positive `usize` values and
    /// otherwise retain the shipped constants. These controls may change how
    /// much work closes before a deadline, so measurement provenance records
    /// their exact raw launch spellings separately.
    #[test]
    fn screen_schedule_env_parsing() {
        let _env = ENV_LOCK.lock().unwrap();
        // Serialized + restored via the blessed env choke point (clippy env
        // wall).
        ny_test_utils::env::with_env_edits(|env| {
            for key in [
                "NY_SCREEN_WAVE_SIZE",
                "NY_SCREEN_CELL_CHUNK",
                "NY_SCREEN_MVF_CHUNK",
            ] {
                env.remove(key);
            }
            assert_eq!(wave_size(), WAVE_SIZE);
            assert_eq!(cell_chunk(), F64_BATCH_CELL_CHUNK);
            assert_eq!(mvf_chunk(), F64_BATCH_MVF_CHUNK);

            env.set("NY_SCREEN_WAVE_SIZE", " 17 ");
            env.set("NY_SCREEN_CELL_CHUNK", "0");
            env.set("NY_SCREEN_MVF_CHUNK", "not-a-number");
            assert_eq!(wave_size(), 17);
            assert_eq!(cell_chunk(), F64_BATCH_CELL_CHUNK);
            assert_eq!(mvf_chunk(), F64_BATCH_MVF_CHUNK);
        });
    }

    /// Batched-wave gating (#f64-batch-boxes): thin-Linear graphs never
    /// batch (measured LOSS vs per-box rayon at 128-wide mscn weights);
    /// fat-Linear graphs batch by default; NY_F64_BATCH_BOXES=0 disables
    /// even those. The screen still verifies through the byte-identical
    /// per-box walks either way.
    #[test]
    fn refine_batch_gate_and_kill_switch() {
        let _env = ENV_LOCK.lock().unwrap();
        // Plateau graph: NO Linear at all — the fat-Linear worthwhile gate
        // keeps the batched lane off, and the per-box lane still proves the
        // clause.
        let graph = build_plateau_graph();
        assert!(!graph.f64_batch_worthwhile());
        let model = BetaCrownModel::Graph(Box::new(graph));
        let input = boxed(vec![0.0, 0.0], vec![2.0, 2.0]);
        let (clauses, boxes) = plateau_clause_and_box();
        let esc = f64_leaf_escalation(&model).expect("escalation available");
        assert!(!esc.batch, "thin graphs must keep the per-box lane");
        let verified = refine_clause_boxes(&model, &input, &clauses, &boxes, None, None);
        assert_eq!(
            verified,
            vec![true],
            "per-box lane must prove the plateau clause"
        );

        // Fat-Linear graph (>= 2^20 params): batch default-ON; kill-switch
        // wins.
        let dim = 1024usize;
        let w = ndarray::Array2::<f32>::from_elem((dim, dim), 0.001);
        let mut fat = ny_propagate::GraphNetwork::new();
        fat.add_node(ny_propagate::GraphNode::from_input(
            "lin",
            ny_propagate::Layer::Linear(ny_propagate::layers::LinearLayer::new(w, None).unwrap()),
        ));
        fat.set_output("lin");
        assert!(fat.f64_batch_worthwhile());
        let fat_model = BetaCrownModel::Graph(Box::new(fat));
        let esc = f64_leaf_escalation(&fat_model).expect("escalation available");
        assert!(esc.batch, "fat-Linear graphs must batch by default");
        // (Serialized + restored via the blessed env choke point.)
        let esc_off =
            ny_test_utils::env::with_serialized_env_vars(&[("NY_F64_BATCH_BOXES", "0")], || {
                f64_leaf_escalation(&fat_model).expect("escalation available")
            });
        assert!(!esc_off.batch, "kill-switch must disable the batched lane");
    }

    /// Fail-closed: a Graph model containing an op the f64 forward does not
    /// support must not build an escalation context.
    #[test]
    fn f64_leaf_escalation_unavailable_for_unsupported_ops() {
        let mut graph = ny_propagate::GraphNetwork::new();
        graph.add_node(ny_propagate::GraphNode::from_input(
            "t",
            ny_propagate::Layer::Tanh(ny_propagate::layers::TanhLayer),
        ));
        graph.set_output("t");
        let model = BetaCrownModel::Graph(Box::new(graph));
        assert!(f64_leaf_escalation(&model).is_none());
    }

    /// End-to-end through the batched FUSED lane (#f64-fused-walk): a
    /// batch-worthy fat-Linear monotone net (2^20-param Linear) with two
    /// band clauses whose margins need f64 — the wave partition routes both
    /// clause boxes through `propagate_ibp_f64_centered_mono_cells_cached`
    /// (2 wide axes each, live clauses) and the screen must verify them.
    /// With NY_F64_BATCH_BOXES=0 the per-box fused lane must produce the
    /// same verdicts (the batched lane is a pure evaluation-plan change).
    #[test]
    fn refine_verifies_band_clauses_through_batched_fused_lane() {
        let _env = ENV_LOCK.lock().unwrap();
        let dim = 1024usize;
        // Exactly representable weight (2^-10): the f64 test oracle below
        // is exact up to f64 dot rounding.
        let wv = 0.0009765625f32;
        let w1 = ndarray::Array2::<f32>::from_elem((dim, dim), wv);
        let w2 = ndarray::Array2::<f32>::from_elem((1, dim), wv);
        let mut g = ny_propagate::GraphNetwork::new();
        g.add_node(ny_propagate::GraphNode::from_input(
            "l1",
            ny_propagate::Layer::Linear(ny_propagate::layers::LinearLayer::new(w1, None).unwrap()),
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "relu",
            ny_propagate::Layer::ReLU(ny_propagate::layers::ReLULayer),
            vec!["l1".to_string()],
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "out",
            ny_propagate::Layer::Linear(ny_propagate::layers::LinearLayer::new(w2, None).unwrap()),
            vec!["relu".to_string()],
        ));
        g.set_output("out");
        assert!(g.f64_batch_worthwhile(), "test net must be batch-worthy");
        assert!(g.supports_ibp_f64_centered());
        let model = BetaCrownModel::Graph(Box::new(g));
        let esc = f64_leaf_escalation(&model).expect("escalation available");
        assert!(esc.batch, "test net must take the batched lane");

        let input = boxed(vec![0.0; dim], vec![1.0; dim]);
        // Two clauses, each pinning all but TWO axes to points (so the
        // fused-lane wide-axis gate admits them) with a band threshold 1e-6
        // above the true maximum: y = 2^-10 · relu-sum with all-positive
        // weights is monotone, so the max sits at the hi corner.
        let mut clauses = Vec::new();
        let mut boxes = Vec::new();
        for (a0, a1) in [(3usize, 700usize), (11usize, 512usize)] {
            let mut b = std::collections::BTreeMap::new();
            let mut hi_sum = 0.0f64;
            for i in 0..dim {
                let (l, h) = if i == a0 || i == a1 {
                    (0.25f64, 0.75f64)
                } else {
                    let p = f64::from(0.1 + (i % 7) as f32 * 0.1);
                    (p, p)
                };
                b.insert(i, (l, h));
                hi_sum += h;
            }
            // y_max = 2^-10 · Σ_h relu(2^-10 · Σ_i x_i) summed over 1024
            // hidden units = 2^-10 · 1024 · (2^-10 · hi_sum) = 2^-10·hi_sum.
            let y_max = 0.0009765625f64 * hi_sum;
            clauses.push(vec![OutputConstraint::GreaterEqConst(0, y_max + 1e-6)]);
            boxes.push(b);
        }

        let verified = refine_clause_boxes(&model, &input, &clauses, &boxes, None, None);
        assert_eq!(
            verified,
            vec![true, true],
            "batched fused lane must prove both band clauses"
        );

        // Kill-switch parity: the per-box fused lane reaches the same
        // verdicts (bounds identical up to kernel selection, both sound).
        // (Serialized + restored via the blessed env choke point.)
        let verified_off =
            ny_test_utils::env::with_serialized_env_vars(&[("NY_F64_BATCH_BOXES", "0")], || {
                refine_clause_boxes(&model, &input, &clauses, &boxes, None, None)
            });
        assert_eq!(
            verified_off,
            vec![true, true],
            "per-box lane must reach the same verdicts"
        );
    }
}

#[cfg(test)]
mod dual_tail_spec_tests {
    use super::*;

    #[test]
    fn clause_to_tail_spec_directions_and_rounding() {
        // Y_0 <= 0.3 (impossible iff l_0 > 0.3): +e_0, t >= 0.3 as f64.
        // Y_1 >= 0.7 (impossible iff u_1 < 0.7): -e_1, t >= -0.7 as f64.
        // Y_0 <= Y_2: e_0 - e_2, t = 0.
        let clause = vec![
            OutputConstraint::LessEqConst(0, 0.3),
            OutputConstraint::GreaterEqConst(1, 0.7),
            OutputConstraint::LessEq(0, 2),
        ];
        let (spec, ths, sizes) = clause_to_tail_spec(&clause, 3).expect("convertible");
        assert_eq!(sizes, vec![3]);
        assert_eq!(spec.shape(), &[3, 3]);
        assert_eq!(spec.row(0).to_vec(), vec![1.0, 0.0, 0.0]);
        assert_eq!(spec.row(1).to_vec(), vec![0.0, -1.0, 0.0]);
        assert_eq!(spec.row(2).to_vec(), vec![1.0, 0.0, -1.0]);
        // Sound threshold rounding: f64(t_f32) must be >= the exact target.
        assert!(f64::from(ths[0]) >= 0.3);
        assert!(f64::from(ths[1]) >= -0.7);
        assert_eq!(ths[2], 0.0);
        // Out-of-range index or exotic constraint: whole clause skips.
        assert!(clause_to_tail_spec(&[OutputConstraint::LessEqConst(9, 0.0)], 3).is_none());
    }

    #[test]
    fn f32_at_least_is_an_upper_f32() {
        for &x in &[0.3_f64, -0.7, 1.0e-7, -1.0e-7, 12345.678, 0.0] {
            let c = f32_at_least(x);
            assert!(f64::from(c) >= x, "f32_at_least({x}) = {c} fell below");
        }
    }
}
