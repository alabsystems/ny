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

use ny_core::{f32_to_f64_exact, f64_to_f32_up, GemmEngine, NyError};
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{
    GraphNetwork, Interval64, Layer, MvfAffineDiagnosticBudget, MvfAffineEnclosure, NETWORK_INPUT,
};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
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

/// Maximum wave admitted to the expensive batched fused f64 MVF lane when a
/// finite verifier deadline is active.
///
/// A 240-root nn4sys dual wave previously entered two indivisible 96-box MVF
/// chunks before ANY root outcome was adjudicated.  The chunks consumed the
/// whole proof slice, then every `bound_node` observed the expired deadline
/// and conservatively discarded its already-computed result.  Eight boxes is
/// the measured latency/progress knee on the 2048-wide dual model: it lets a
/// completed chunk discharge/split roots before the next expensive walk while
/// retaining enough stacked rows for the batched Linear kernels.
///
/// This cap is deliberately model-, property-, lane-, and deadline-specific.
/// Non-dual MSCN, the 128-wide dual, cheap cell-only, non-MVF, and
/// deadline-free screens retain the historical `WAVE_SIZE`.
const F64_BATCH_MVF_DEADLINE_WAVE: usize = 8;

fn mvf_deadline_wave_size() -> usize {
    env_usize("NY_SCREEN_MVF_WAVE_SIZE", F64_BATCH_MVF_DEADLINE_WAVE)
}

fn scheduled_wave_len(
    queued: usize,
    f64_leaf: Option<&F64LeafEscalation<'_>>,
    deadline: Option<Instant>,
    exact_mscn_2048_dual: bool,
) -> usize {
    let cap = if exact_mscn_2048_dual
        && deadline.is_some()
        && f64_leaf.is_some_and(|esc| esc.batch && esc.mvf)
    {
        wave_size().min(mvf_deadline_wave_size())
    } else {
        wave_size()
    };
    queued.min(cap)
}

/// Whether a completed parallel wave crossed the verifier deadline and must
/// be discarded as one unit.  Applying only the workers that happened to
/// finish before a post-join observation would make the queue trajectory
/// scheduler-dependent; requeueing the whole wave keeps the transition
/// atomic and conservative.
fn completed_wave_must_be_discarded(deadline: Option<Instant>, observed_at: Instant) -> bool {
    deadline.is_some_and(|deadline| observed_at >= deadline)
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

/// Schedule-only opt-in for an exact-cover-seeded NN4SYS closer. Independently
/// authenticated clauses are removed before the mature screen is built; on
/// the authentic 128d dual pilot, its remaining first wave launched 24
/// parallel CROWN probes (~428ms each) before the adaptive gate could latch,
/// while the f64 cell/MVF lane supplied every published clause certificate.
/// Pre-shedding can only omit an optional tightening attempt: unresolved boxes
/// remain in the sound f64 split queue, so this has no proof authority.
fn seeded_crown_preshed_enabled_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

fn seeded_crown_preshed_enabled() -> bool {
    seeded_crown_preshed_enabled_from_value(
        std::env::var("NY_NN4SYS_SEEDED_SHED_CROWN").ok().as_deref(),
    )
}

/// Verdict-neutral nn4sys one-dimensional phase-event diagnostic. Default
/// OFF: no point-JVP walks or telemetry allocations occur unless explicitly
/// requested. Candidate events never feed the split queue or any bound.
fn phase_event_diag_enabled_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

fn phase_event_diag_enabled() -> bool {
    phase_event_diag_enabled_from_value(std::env::var("NY_NN4SYS_1D_PHASE_EVENTS").ok().as_deref())
}

/// Keep explicit diagnostic runs bounded on the larger (10k-clause) rows.
const PHASE_EVENT_DIAG_MAX_GROUPS: usize = 256;
const PHASE_EVENT_DIAG_HARD_MAX_GROUPS: usize = 4096;
const PHASE_EVENT_DIAG_HARD_BUDGET: Duration = Duration::from_millis(100);
const PHASE_EVENT_DIAG_OUTER_RESERVE_DIVISOR: u32 = 20;
const NN4SYS_PHASE_EVENT_TELEMETRY_MARKER: &str = "NY_NN4SYS_1D_PHASE_EVENTS_V1";

fn phase_event_diag_max_groups_from_value(value: Option<&str>) -> usize {
    value
        .filter(|raw| !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&parsed| (1..=PHASE_EVENT_DIAG_HARD_MAX_GROUPS).contains(&parsed))
        .unwrap_or(PHASE_EVENT_DIAG_MAX_GROUPS)
}

fn phase_event_diag_max_groups() -> usize {
    phase_event_diag_max_groups_from_value(
        std::env::var("NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS")
            .ok()
            .as_deref(),
    )
}

/// Winner-derived MVF-Clip M0 diagnostic. The exact gate is default OFF and
/// only exposes telemetry from already-paid batched MVF derivative channels.
/// It never edits a clause box or feeds a proof decision.
fn mvf_clip_diag_enabled_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

fn mvf_clip_diag_enabled() -> bool {
    mvf_clip_diag_enabled_from_value(std::env::var("NY_NN4SYS_MVF_CLIP_DIAG").ok().as_deref())
}

const MVF_CLIP_DIAG_MAX_GROUPS: usize = 256;
const MVF_CLIP_DIAG_HARD_MAX_GROUPS: usize = 4096;
const MVF_CLIP_DIAG_MAX_SAMPLES: usize = 1024;
const MVF_CLIP_DIAG_HARD_MAX_SAMPLES: usize = 16_384;
const MVF_CLIP_DIAG_AFFINE_BUDGET: Duration = Duration::from_millis(10);
const MVF_CLIP_DIAG_REDUCTION_BUDGET: Duration = Duration::from_millis(20);
const NN4SYS_MVF_CLIP_TELEMETRY_MARKER: &str = "NY_NN4SYS_MVF_CLIP_DIAG_V1";

fn bounded_positive_decimal_from_value(
    value: Option<&str>,
    default: usize,
    hard_max: usize,
) -> usize {
    value
        .filter(|raw| !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&parsed| (1..=hard_max).contains(&parsed))
        .unwrap_or(default)
}

fn mvf_clip_diag_max_groups_from_value(value: Option<&str>) -> usize {
    bounded_positive_decimal_from_value(
        value,
        MVF_CLIP_DIAG_MAX_GROUPS,
        MVF_CLIP_DIAG_HARD_MAX_GROUPS,
    )
}

fn mvf_clip_diag_max_groups() -> usize {
    mvf_clip_diag_max_groups_from_value(
        std::env::var("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS")
            .ok()
            .as_deref(),
    )
}

fn mvf_clip_diag_max_samples_from_value(value: Option<&str>) -> usize {
    bounded_positive_decimal_from_value(
        value,
        MVF_CLIP_DIAG_MAX_SAMPLES,
        MVF_CLIP_DIAG_HARD_MAX_SAMPLES,
    )
}

fn mvf_clip_diag_max_samples() -> usize {
    mvf_clip_diag_max_samples_from_value(
        std::env::var("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES")
            .ok()
            .as_deref(),
    )
}

/// Give the dark diagnostic at most 100 ms and at most 5% of the verifier's
/// remaining wall-clock budget. Thus at least 95% of the observed remaining
/// time is reserved for proof-producing work.
fn phase_event_call_deadline(outer_deadline: Option<Instant>) -> (Instant, u128) {
    let now = Instant::now();
    let budget = outer_deadline
        .map(|deadline| {
            deadline
                .saturating_duration_since(now)
                .checked_div(PHASE_EVENT_DIAG_OUTER_RESERVE_DIVISOR)
                .unwrap_or(Duration::ZERO)
                .min(PHASE_EVENT_DIAG_HARD_BUDGET)
        })
        .unwrap_or(PHASE_EVENT_DIAG_HARD_BUDGET);
    (now.checked_add(budget).unwrap_or(now), budget.as_millis())
}

/// Smallest f32 at or above `x` — thresholds handed to the f64 tail are
/// decoded exactly as f64, so rounding UP keeps `l > t` at least as strong as
/// the exact `l > x` impossibility test (sound, conservative). The shared
/// converter is bit-classified and does not depend on FTZ/DAZ.
fn f32_at_least(x: f64) -> f32 {
    f64_to_f32_up(x)
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

    /// Deterministically shed the in-node CROWN speed heuristic for the
    /// finite-deadline MSCN-dual MVF pilot. Its one adaptive probe is measured
    /// to close nothing and, under a capped Rayon wave, whichever node wins
    /// that probe is scheduler-dependent. Proof semantics are unchanged:
    /// skipped rows remain open for the sound f64 bounds/splits.
    ///
    /// An explicit `NY_SCREEN_CROWN_MS=0` remains authoritative: it disables
    /// the cost gate, so this deterministic pilot policy becomes a no-op.
    fn force_shed(&mut self) {
        if self.threshold_ms.is_some() {
            self.shed.store(true, Ordering::Relaxed);
        }
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

type ClauseBoxGroups = HashMap<Vec<u32>, (BoundedTensor, Vec<usize>)>;

/// Convert bit-keyed root groups to the initial DFS stack in deterministic
/// key order.  The refinement loop pops from the tail, so this yields a stable
/// descending-key root order (and stable capped-wave clause selection) across
/// processes and hash seeds.
fn ordered_root_queue(groups: ClauseBoxGroups) -> Vec<BoxNode> {
    let mut groups: Vec<_> = groups.into_iter().collect();
    groups.sort_unstable_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    groups
        .into_iter()
        .map(|(_, (bounds, active))| {
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
        .collect()
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
    graph: &'a GraphNetwork,
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

/// The exact unique axis selected by the production centered-form absorption
/// rule, or `None` unless precisely one axis is seeded.
fn single_centered_seed_axis(bounds: &BoundedTensor) -> Option<usize> {
    let lo = bounds.lower().as_slice()?;
    let hi = bounds.upper().as_slice()?;
    let axes = ny_propagate::centered_seed_axis_indices_f32(lo, hi);
    let &[axis] = axes.as_slice() else {
        return None;
    };
    Some(axis)
}

/// Aggregate from the dark point-JVP phase-event probe. This struct is not
/// consulted by the verifier; its only consumer is the telemetry log.
#[derive(Debug, Default, PartialEq)]
struct PhaseEventScreenTelemetry {
    target_eligible: bool,
    property_census_complete: bool,
    property_clauses: usize,
    property_dim_0: usize,
    property_dim_1: usize,
    property_dim_2: usize,
    property_dim_3: usize,
    property_dim_4: usize,
    property_dim_5: usize,
    property_dim_overflow: usize,
    eligible_groups: usize,
    probed_groups: usize,
    capped_groups: usize,
    deadline_capped_groups: usize,
    declined_groups: usize,
    collector_errors: usize,
    deadline_errors: usize,
    deadline_hit: bool,
    relu_nodes: usize,
    preactivations: usize,
    candidates: usize,
    at_point_kinks: usize,
    stationary: usize,
    slope_straddles_zero: usize,
    non_finite: usize,
    outside_scan: usize,
    candidates_p50: usize,
    candidates_p95: usize,
    candidates_max: usize,
    budget_ms: u128,
    elapsed_ms: u128,
}

fn phase_event_telemetry_line_if(
    enabled: bool,
    phase: &PhaseEventScreenTelemetry,
) -> Option<String> {
    enabled.then(|| {
        format!(
            "{NN4SYS_PHASE_EVENT_TELEMETRY_MARKER} \
target_eligible={} property_census_complete={} property_clauses={} \
property_dim_0={} property_dim_1={} property_dim_2={} property_dim_3={} \
property_dim_4={} property_dim_5={} property_dim_overflow={} \
eligible_groups={} attempted_groups={} cap_skipped_groups={} \
deadline_skipped_groups={} declined_groups={} collector_errors={} \
deadline_errors={} deadline_hit={} relu_nodes={} preactivations={} \
candidates={} at_point_kinks={} stationary={} slope_straddles_zero={} \
non_finite={} outside_scan={} candidates_p50={} candidates_p95={} \
candidates_max={} budget_ms={} elapsed_ms={} verdict_authority=false",
            phase.target_eligible,
            phase.property_census_complete,
            phase.property_clauses,
            phase.property_dim_0,
            phase.property_dim_1,
            phase.property_dim_2,
            phase.property_dim_3,
            phase.property_dim_4,
            phase.property_dim_5,
            phase.property_dim_overflow,
            phase.eligible_groups,
            phase.probed_groups,
            phase.capped_groups,
            phase.deadline_capped_groups,
            phase.declined_groups,
            phase.collector_errors,
            phase.deadline_errors,
            phase.deadline_hit,
            phase.relu_nodes,
            phase.preactivations,
            phase.candidates,
            phase.at_point_kinks,
            phase.stationary,
            phase.slope_straddles_zero,
            phase.non_finite,
            phase.outside_scan,
            phase.candidates_p50,
            phase.candidates_p95,
            phase.candidates_max,
            phase.budget_ms,
            phase.elapsed_ms,
        )
    })
}

#[cfg(test)]
thread_local! {
    static PHASE_EVENT_TEST_ATTEMPTS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

const MSCN_DUAL_MAX_GRAPH_NODES: usize = 128;
const MSCN_DUAL_MAX_GRAPH_EDGES: usize = 512;
const MSCN_DUAL_MAX_NODE_INPUTS: usize = 64;

// Exact SHA-256 allowlist of the deterministic post-loader graph structures
// below. The signature includes loader execution order, ordered edges, every
// supported layer's structural parameters, and inferred input/output shapes;
// weights and node-name bytes are intentionally excluded.
const MSCN_128D_DUAL_STRUCTURAL_SHA256: &str =
    "1aafab11e0b169078736691a91333f2333d6aa4a32180f0e9dcee7380bae0e55";
const MSCN_2048D_DUAL_STRUCTURAL_SHA256: &str =
    "27fa023c681dcedf0258f9482884ab51777d342f46bbfb46a49ea7dd85804c47";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MscnDualVariant {
    D128,
    D2048,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum StructuralInput {
    NetworkInput,
    Node(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StructuralNode {
    layer: String,
    inputs: Vec<StructuralInput>,
    output_shape: Option<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MscnDualStructure {
    input_shape: Option<Vec<usize>>,
    nodes: Vec<StructuralNode>,
    output: usize,
}

fn mscn_dual_layer_descriptor(layer: &Layer) -> Option<String> {
    match layer {
        Layer::Slice(layer) => Some(format!(
            "slice(axis={},start={},end={})",
            layer.axis, layer.start, layer.end
        )),
        Layer::ReLU(_) => Some("relu".to_string()),
        Layer::AddConstant(layer) => Some(format!(
            "add_constant(shape={:?})",
            layer.constant().shape()
        )),
        Layer::ReduceSum(layer) => Some(format!(
            "reduce_sum(axes={:?},keepdims={})",
            layer.axes, layer.keepdims
        )),
        Layer::MulBinary(_) => Some("mul_binary".to_string()),
        Layer::Div(_) => Some("div".to_string()),
        Layer::Linear(layer) => Some(format!(
            "linear(in={},out={},bias={})",
            layer.in_features(),
            layer.out_features(),
            layer.bias().is_some()
        )),
        Layer::Concat(layer) => Some(format!(
            "concat(axis={},input_shapes={:?})",
            layer.axis, layer.input_shapes
        )),
        Layer::Sigmoid(_) => Some("sigmoid".to_string()),
        Layer::Sub(_) => Some("sub".to_string()),
        _ => None,
    }
}

fn phase_event_gate_deadline_check(deadline: Option<Instant>) -> Result<(), ()> {
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        Err(())
    } else {
        Ok(())
    }
}

/// Extract a bounded, deterministic topology record from the loaded DAG.
///
/// The cheap public node/edge counts are checked before `exec_order`, so a
/// non-target graph cannot trigger an unbounded topological prepass.
/// Unsupported ops, dangling/forward edges, oversized graphs, or missing
/// outputs fail closed. Node names are not hashed, but their deterministic
/// loader ordering is part of this exact-fixture contract.
fn mscn_dual_structure_before(
    graph: &GraphNetwork,
    deadline: Option<Instant>,
) -> Result<Option<MscnDualStructure>, ()> {
    phase_event_gate_deadline_check(deadline)?;
    if graph.num_nodes() == 0
        || graph.num_nodes() > MSCN_DUAL_MAX_GRAPH_NODES
        || graph.node_names().len() != graph.num_nodes()
    {
        return Ok(None);
    }
    let mut edge_count = 0usize;
    for name in graph.node_names() {
        phase_event_gate_deadline_check(deadline)?;
        let Some(node) = graph.node(name) else {
            return Ok(None);
        };
        if node.inputs().len() > MSCN_DUAL_MAX_NODE_INPUTS {
            return Ok(None);
        }
        let Some(next_edges) = edge_count.checked_add(node.inputs().len()) else {
            return Ok(None);
        };
        edge_count = next_edges;
        if edge_count > MSCN_DUAL_MAX_GRAPH_EDGES {
            return Ok(None);
        }
    }
    phase_event_gate_deadline_check(deadline)?;
    let Ok(order) = graph.exec_order() else {
        return Ok(None);
    };
    if order.len() != graph.num_nodes() {
        return Ok(None);
    }
    let mut ordinals = HashMap::with_capacity(order.len());
    let mut nodes = Vec::with_capacity(order.len());
    for (ordinal, name) in order.iter().enumerate() {
        phase_event_gate_deadline_check(deadline)?;
        let Some(node) = graph.node(name) else {
            return Ok(None);
        };
        let Some(layer) = mscn_dual_layer_descriptor(node.layer()) else {
            return Ok(None);
        };
        let mut inputs = Vec::with_capacity(node.inputs().len());
        for input in node.inputs() {
            if input == NETWORK_INPUT {
                inputs.push(StructuralInput::NetworkInput);
            } else {
                let Some(&parent) = ordinals.get(input.as_str()) else {
                    return Ok(None);
                };
                inputs.push(StructuralInput::Node(parent));
            }
        }
        nodes.push(StructuralNode {
            layer,
            inputs,
            output_shape: graph.declared_shape(name).map(<[usize]>::to_vec),
        });
        ordinals.insert(name.as_str(), ordinal);
    }
    let Some(&output) = ordinals.get(graph.output_name()) else {
        return Ok(None);
    };
    Ok(Some(MscnDualStructure {
        input_shape: graph.declared_shape(NETWORK_INPUT).map(<[usize]>::to_vec),
        nodes,
        output,
    }))
}

#[cfg(all(test, feature = "external-vnncomp"))]
fn mscn_dual_structure(graph: &GraphNetwork) -> Option<MscnDualStructure> {
    mscn_dual_structure_before(graph, None).ok().flatten()
}

fn hash_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn hash_shape(hasher: &mut Sha256, shape: Option<&[usize]>) {
    match shape {
        Some(shape) => {
            hasher.update([1]);
            hash_usize(hasher, shape.len());
            for &dim in shape {
                hash_usize(hasher, dim);
            }
        }
        None => hasher.update([0]),
    }
}

fn mscn_dual_structure_digest_before(
    structure: &MscnDualStructure,
    deadline: Option<Instant>,
) -> Result<String, ()> {
    phase_event_gate_deadline_check(deadline)?;
    let mut hasher = Sha256::new();
    hasher.update(b"NY_MSCN_DUAL_TOPOLOGY_V1\0");
    hash_shape(&mut hasher, structure.input_shape.as_deref());
    hash_usize(&mut hasher, structure.nodes.len());
    for node in &structure.nodes {
        phase_event_gate_deadline_check(deadline)?;
        hash_usize(&mut hasher, node.layer.len());
        hasher.update(node.layer.as_bytes());
        hash_usize(&mut hasher, node.inputs.len());
        for input in &node.inputs {
            match input {
                StructuralInput::NetworkInput => hasher.update([0]),
                StructuralInput::Node(ordinal) => {
                    hasher.update([1]);
                    hash_usize(&mut hasher, *ordinal);
                }
            }
        }
        hash_shape(&mut hasher, node.output_shape.as_deref());
    }
    hash_usize(&mut hasher, structure.output);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(all(test, feature = "external-vnncomp"))]
fn mscn_dual_structure_digest(structure: &MscnDualStructure) -> String {
    mscn_dual_structure_digest_before(structure, None).expect("deadline-free structural digest")
}

fn classify_mscn_dual_digest(digest: &str) -> Option<MscnDualVariant> {
    match digest {
        MSCN_128D_DUAL_STRUCTURAL_SHA256 => Some(MscnDualVariant::D128),
        MSCN_2048D_DUAL_STRUCTURAL_SHA256 => Some(MscnDualVariant::D2048),
        _ => None,
    }
}

fn classify_nn4sys_mscn_dual_graph_before(
    graph: &GraphNetwork,
    deadline: Option<Instant>,
) -> Result<Option<MscnDualVariant>, ()> {
    let Some(structure) = mscn_dual_structure_before(graph, deadline)? else {
        return Ok(None);
    };
    let digest = mscn_dual_structure_digest_before(&structure, deadline)?;
    Ok(classify_mscn_dual_digest(&digest))
}

#[cfg(test)]
fn is_nn4sys_mscn_dual_graph(graph: &GraphNetwork) -> bool {
    classify_nn4sys_mscn_dual_graph_before(graph, None)
        .ok()
        .flatten()
        .is_some()
}

const MSCN_DUAL_PROPERTY_CARDINALITIES: &[usize] = &[
    1, 240, 360, 480, 600, 720, 840, 960, 2_260, 2_890, 3_520, 4_150, 4_780, 5_410, 6_040, 6_670,
    7_300, 7_930, 8_560, 9_190, 9_820, 10_450, 11_080,
];

fn has_exact_mscn_dual_clause_surface(
    clause: &std::collections::BTreeMap<usize, (f64, f64)>,
) -> bool {
    clause.len() == 22 * 14
        && clause.first_key_value().is_some_and(|(&axis, _)| axis == 0)
        && clause
            .last_key_value()
            .is_some_and(|(&axis, _)| axis == 22 * 14 - 1)
}

/// Exact diagnostic admission: known MSCN-dual structural signature plus every
/// clause's full 308-coordinate box and matching runtime tensor surface.
#[cfg(all(test, feature = "external-vnncomp"))]
fn phase_event_target_eligible(
    model_net: &BetaCrownModel,
    groups: &HashMap<Vec<u32>, (BoundedTensor, Vec<usize>)>,
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
) -> bool {
    phase_event_target_eligible_before(model_net, groups, per_clause_input_bounds, None)
        .unwrap_or(false)
}

/// Deadline-aware exact admission prepass. `Err(())` means the diagnostic
/// budget expired, never that the proof failed.
fn phase_event_target_eligible_before(
    model_net: &BetaCrownModel,
    groups: &HashMap<Vec<u32>, (BoundedTensor, Vec<usize>)>,
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    deadline: Option<Instant>,
) -> Result<bool, ()> {
    let expired = || deadline.is_some_and(|deadline| Instant::now() >= deadline);
    if expired() {
        return Err(());
    }
    let BetaCrownModel::Graph(graph) = model_net else {
        return Ok(false);
    };
    let clause_count = per_clause_input_bounds.len();
    if !MSCN_DUAL_PROPERTY_CARDINALITIES.contains(&clause_count)
        || groups.is_empty()
        || groups.len() > clause_count
    {
        return Ok(false);
    }
    if classify_nn4sys_mscn_dual_graph_before(graph, deadline)?.is_none() {
        return Ok(false);
    }
    for clause in per_clause_input_bounds {
        if expired() {
            return Err(());
        }
        if !has_exact_mscn_dual_clause_surface(clause) {
            return Ok(false);
        }
    }
    for (bounds, active) in groups.values() {
        if expired() {
            return Err(());
        }
        if bounds.lower().len() != 22 * 14
            || bounds.upper().len() != 22 * 14
            || active.is_empty()
            || active.iter().any(|&clause| clause >= clause_count)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Production admission for the small finite-deadline MVF wave. Keep this
/// narrower than the diagnostic's two-model gate: the 2048-wide MSCN dual is
/// the measured starvation case, while non-dual MSCN and the 128-wide dual
/// retain their high-throughput historical waves.
fn exact_mscn_2048_dual_screen_before(
    model_net: &BetaCrownModel,
    groups: &HashMap<Vec<u32>, (BoundedTensor, Vec<usize>)>,
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    deadline: Option<Instant>,
) -> Result<bool, ()> {
    if !phase_event_target_eligible_before(model_net, groups, per_clause_input_bounds, deadline)? {
        return Ok(false);
    }
    let BetaCrownModel::Graph(graph) = model_net else {
        return Ok(false);
    };
    Ok(matches!(
        classify_nn4sys_mscn_dual_graph_before(graph, deadline)?,
        Some(MscnDualVariant::D2048)
    ))
}

/// Build an exact point box at the f64 midpoint of a clause box when the
/// production sectioned-centering rule sees exactly one genuine wide axis.
fn phase_event_probe_point(bounds: &BoundedTensor) -> Option<(Interval64, usize, f64, f64)> {
    let lo_std = bounds.lower().as_standard_layout();
    let hi_std = bounds.upper().as_standard_layout();
    let (lo, hi) = (lo_std.as_slice()?, hi_std.as_slice()?);
    if lo.len() != hi.len() {
        return None;
    }
    let axis = single_centered_seed_axis(bounds)?;
    let mut point = Vec::with_capacity(lo.len());
    for (&l, &h) in lo.iter().zip(hi) {
        if !(l.is_finite() && h.is_finite() && l <= h) {
            return None;
        }
        let l = f32_to_f64_exact(l);
        let h = f32_to_f64_exact(h);
        point.push(f64::midpoint(l, h).clamp(l, h));
    }
    let shape = bounds.lower().shape().to_vec();
    let point = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), point).ok()?;
    Some((
        Interval64::point(point),
        axis,
        f32_to_f64_exact(lo[axis]),
        f32_to_f64_exact(hi[axis]),
    ))
}

fn phase_event_quantile(sorted: &[usize], numerator: usize, denominator: usize) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(sorted.len() - 1).saturating_mul(numerator) / denominator]
}

/// Run the dark one-dimensional point-JVP probe over root groups.
///
/// The group map and verifier state are borrowed immutably. The returned
/// counters are logged and then dropped; no candidate endpoint can reach
/// `BoxNode`, `NodeOutcome`, or the final result vector.
fn collect_phase_event_screen_telemetry(
    model_net: &BetaCrownModel,
    groups: &HashMap<Vec<u32>, (BoundedTensor, Vec<usize>)>,
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    max_groups: usize,
    call_deadline: Instant,
    budget_ms: u128,
    target_eligible_override: Option<bool>,
) -> PhaseEventScreenTelemetry {
    let started = Instant::now();
    let mut out = PhaseEventScreenTelemetry {
        property_clauses: per_clause_input_bounds.len(),
        budget_ms,
        ..PhaseEventScreenTelemetry::default()
    };

    out.target_eligible = match target_eligible_override {
        Some(target_eligible) => target_eligible,
        None => match phase_event_target_eligible_before(
            model_net,
            groups,
            per_clause_input_bounds,
            Some(call_deadline),
        ) {
            Ok(target_eligible) => target_eligible,
            Err(()) => {
                out.deadline_hit = true;
                out.deadline_capped_groups = groups.len();
                out.elapsed_ms = started.elapsed().as_millis();
                return out;
            }
        },
    };
    if !out.target_eligible {
        out.declined_groups = groups.len();
        out.elapsed_ms = started.elapsed().as_millis();
        return out;
    }

    for clause in per_clause_input_bounds {
        if Instant::now() >= call_deadline {
            out.deadline_hit = true;
            out.deadline_capped_groups = groups.len();
            out.elapsed_ms = started.elapsed().as_millis();
            return out;
        }
        let dims = clause.values().filter(|&&(l, u)| u > l).count();
        match dims {
            0 => out.property_dim_0 += 1,
            1 => out.property_dim_1 += 1,
            2 => out.property_dim_2 += 1,
            3 => out.property_dim_3 += 1,
            4 => out.property_dim_4 += 1,
            5 => out.property_dim_5 += 1,
            _ => out.property_dim_overflow += 1,
        }
    }
    out.property_census_complete = true;

    let BetaCrownModel::Graph(graph) = model_net else {
        out.declined_groups = groups.len();
        out.elapsed_ms = started.elapsed().as_millis();
        return out;
    };
    if !graph.supports_ibp_f64_centered() {
        out.declined_groups = groups.len();
        out.elapsed_ms = started.elapsed().as_millis();
        return out;
    }

    // HashMap iteration is intentionally normalized by the first clause id so
    // capped diagnostic runs probe the same roots across processes.
    let mut ordered: Vec<_> = groups.values().collect();
    ordered.sort_by_key(|(_, active)| active.first().copied().unwrap_or(usize::MAX));
    let mut candidate_counts = Vec::new();
    for (group_index, (bounds, _)) in ordered.iter().enumerate() {
        if Instant::now() >= call_deadline {
            out.deadline_hit = true;
            out.deadline_capped_groups += ordered.len() - group_index;
            break;
        }
        let Some((point, axis, scan_lower, scan_upper)) = phase_event_probe_point(bounds) else {
            out.declined_groups += 1;
            continue;
        };
        out.eligible_groups += 1;
        if out.probed_groups >= max_groups {
            out.capped_groups += ordered.len() - group_index;
            break;
        }
        if Instant::now() >= call_deadline {
            out.deadline_hit = true;
            out.deadline_capped_groups += ordered.len() - group_index;
            break;
        }
        // Count attempts, not just successes: malformed/unsupported walks
        // must not let a diagnostic error storm bypass the explicit cap.
        out.probed_groups += 1;
        #[cfg(test)]
        PHASE_EVENT_TEST_ATTEMPTS.with(|attempts| attempts.set(attempts.get() + 1));
        match graph.diagnose_relu_phase_events_1d_until(
            &point,
            axis,
            scan_lower,
            scan_upper,
            call_deadline,
        ) {
            Ok(diag) => {
                out.relu_nodes += diag.relu_nodes;
                out.preactivations += diag.preactivations;
                out.candidates += diag.candidates.len();
                out.at_point_kinks += diag.at_point_kinks;
                out.stationary += diag.stationary;
                out.slope_straddles_zero += diag.slope_straddles_zero;
                out.non_finite += diag.non_finite;
                out.outside_scan += diag.outside_scan;
                candidate_counts.push(diag.candidates.len());
            }
            Err(NyError::DeadlineExceeded(_)) => {
                out.deadline_errors += 1;
                out.deadline_hit = true;
                out.deadline_capped_groups += ordered.len() - group_index - 1;
                break;
            }
            Err(_) => out.collector_errors += 1,
        }
    }
    candidate_counts.sort_unstable();
    out.candidates_p50 = phase_event_quantile(&candidate_counts, 50, 100);
    out.candidates_p95 = phase_event_quantile(&candidate_counts, 95, 100);
    out.candidates_max = candidate_counts.last().copied().unwrap_or(0);
    out.elapsed_ms = started.elapsed().as_millis();
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MvfAffineRow1d {
    remainder_lower: f64,
    remainder_upper: f64,
    slope: f64,
    center: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MvfClauseClip1d {
    kept_lower: f64,
    kept_upper: f64,
    constraints_used: usize,
    unsupported_constraints: usize,
}

impl MvfClauseClip1d {
    fn empty(self) -> bool {
        self.kept_lower > self.kept_upper
    }

    fn kept_ratio(self, domain_lower: f64, domain_upper: f64) -> f64 {
        if self.empty() {
            return 0.0;
        }
        let width = domain_upper - domain_lower;
        if width <= 0.0 {
            return 1.0;
        }
        ((self.kept_upper - self.kept_lower).max(0.0) / width).clamp(0.0, 1.0)
    }
}

/// Build a one-dimensional affine enclosure for a ±1 output row.
///
/// Combining output elements can round the row coefficient. The exact real
/// coefficient is first enclosed outward, a new f64 midpoint is selected, and
/// the coefficient-rounding radius is absorbed into the row remainder. Thus
/// `row(x) ∈ remainder + slope*(x-center)` remains certified.
fn mvf_affine_row_1d(
    affine: &MvfAffineEnclosure,
    terms: &[(usize, i8)],
    domain_lower: f64,
    domain_upper: f64,
) -> Option<MvfAffineRow1d> {
    let [axis] = affine.seed_axes.as_slice() else {
        return None;
    };
    let [center] = affine.centers.as_slice() else {
        return None;
    };
    let [coefficients] = affine.coefficients.as_slice() else {
        return None;
    };
    if !(domain_lower.is_finite()
        && domain_upper.is_finite()
        && domain_lower <= *center
        && *center <= domain_upper)
    {
        return None;
    }
    let rem_lo_std = affine.remainder.lower.as_standard_layout();
    let rem_hi_std = affine.remainder.upper.as_standard_layout();
    let coeff_std = coefficients.as_standard_layout();
    let (rem_lo, rem_hi, coeff) = (
        rem_lo_std.as_slice()?,
        rem_hi_std.as_slice()?,
        coeff_std.as_slice()?,
    );
    if rem_lo.len() != rem_hi.len() || rem_lo.len() != coeff.len() {
        return None;
    }

    let mut row_lo = 0.0f64;
    let mut row_hi = 0.0f64;
    let mut slope_lo = 0.0f64;
    let mut slope_hi = 0.0f64;
    for &(output, sign) in terms {
        if output >= rem_lo.len() || !matches!(sign, -1 | 1) {
            return None;
        }
        let (term_lo, term_hi, term_slope) = if sign == 1 {
            (rem_lo[output], rem_hi[output], coeff[output])
        } else {
            (
                (-rem_hi[output]).next_down(),
                (-rem_lo[output]).next_up(),
                -coeff[output],
            )
        };
        if !(term_lo.is_finite()
            && term_hi.is_finite()
            && term_lo <= term_hi
            && term_slope.is_finite())
        {
            return None;
        }
        row_lo = (row_lo + term_lo).next_down();
        row_hi = (row_hi + term_hi).next_up();
        slope_lo = (slope_lo + term_slope).next_down();
        slope_hi = (slope_hi + term_slope).next_up();
    }
    if !(row_lo.is_finite()
        && row_hi.is_finite()
        && row_lo <= row_hi
        && slope_lo.is_finite()
        && slope_hi.is_finite()
        && slope_lo <= slope_hi)
    {
        return None;
    }

    let slope = f64::midpoint(slope_lo, slope_hi).clamp(slope_lo, slope_hi);
    let radius = (*center - domain_lower)
        .next_up()
        .max((domain_upper - *center).next_up());
    let coefficient_error = (slope - slope_lo)
        .next_up()
        .max((slope_hi - slope).next_up());
    let error_radius = (coefficient_error * radius).next_up();
    row_lo = (row_lo - error_radius).next_down();
    row_hi = (row_hi + error_radius).next_up();
    if !(slope.is_finite()
        && radius.is_finite()
        && coefficient_error.is_finite()
        && error_radius.is_finite()
        && row_lo.is_finite()
        && row_hi.is_finite()
        && row_lo <= row_hi)
    {
        return None;
    }
    let _ = axis; // The caller independently checks the box's unique seed.
    Some(MvfAffineRow1d {
        remainder_lower: row_lo,
        remainder_upper: row_hi,
        slope,
        center: *center,
    })
}

/// Conservative interval where `row(x) <= threshold` might still hold.
///
/// Only the affine LOWER endpoint is needed: if it exceeds the threshold, the
/// unsafe constraint is impossible. Strict constraints intentionally reuse
/// this non-strict relation, retaining boundary points conservatively.
fn mvf_clip_le_row_1d(
    row: MvfAffineRow1d,
    threshold: f64,
    domain_lower: f64,
    domain_upper: f64,
) -> Option<(f64, f64)> {
    if !(threshold.is_finite()
        && domain_lower.is_finite()
        && domain_upper.is_finite()
        && domain_lower <= row.center
        && row.center <= domain_upper)
    {
        return None;
    }
    if row.slope == 0.0 {
        return (row.remainder_lower <= threshold)
            .then_some((domain_lower, domain_upper))
            .or(Some((1.0, 0.0)));
    }

    // Enclose the exact root center + (threshold - remainder_lower)/slope.
    let numerator_lo = (threshold - row.remainder_lower).next_down();
    let numerator_hi = (threshold - row.remainder_lower).next_up();
    let q0 = numerator_lo / row.slope;
    let q1 = numerator_hi / row.slope;
    let root_lower = (row.center + q0.min(q1).next_down()).next_down();
    let root_upper = (row.center + q0.max(q1).next_up()).next_up();
    if !(root_lower.is_finite() && root_upper.is_finite() && root_lower <= root_upper) {
        return None;
    }
    if row.slope > 0.0 {
        Some((domain_lower, domain_upper.min(root_upper)))
    } else {
        Some((domain_lower.max(root_lower), domain_upper))
    }
}

/// Clause-local retained interval under a certified MVF affine enclosure.
///
/// Every supported unsafe constraint clips the SAME clause-local interval by
/// intersection because a clause is a conjunction. Different clauses are
/// evaluated independently, so opposing band tails are never unioned.
fn mvf_clip_clause_1d(
    affine: &MvfAffineEnclosure,
    clause: &[OutputConstraint],
    domain_lower: f64,
    domain_upper: f64,
    deadline: Option<Instant>,
) -> Result<Option<MvfClauseClip1d>, ()> {
    let mut kept_lower = domain_lower;
    let mut kept_upper = domain_upper;
    let mut constraints_used = 0usize;
    let mut unsupported_constraints = 0usize;
    for constraint in clause {
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(());
        }
        let (terms, threshold): (Vec<(usize, i8)>, f64) = match constraint {
            OutputConstraint::LessEqConst(i, k) | OutputConstraint::LessThanConst(i, k) => {
                (vec![(*i, 1)], *k)
            }
            OutputConstraint::GreaterEqConst(i, k) | OutputConstraint::GreaterThanConst(i, k) => {
                (vec![(*i, -1)], -*k)
            }
            OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                (vec![(*i, 1), (*j, -1)], 0.0)
            }
            OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                (vec![(*j, 1), (*i, -1)], 0.0)
            }
            _ => {
                unsupported_constraints += 1;
                continue;
            }
        };
        let Some(row) = mvf_affine_row_1d(affine, &terms, domain_lower, domain_upper) else {
            unsupported_constraints += 1;
            continue;
        };
        let Some((row_lower, row_upper)) =
            mvf_clip_le_row_1d(row, threshold, domain_lower, domain_upper)
        else {
            unsupported_constraints += 1;
            continue;
        };
        constraints_used += 1;
        kept_lower = kept_lower.max(row_lower);
        kept_upper = kept_upper.min(row_upper);
        if kept_lower > kept_upper {
            break;
        }
    }
    Ok((constraints_used > 0).then_some(MvfClauseClip1d {
        kept_lower,
        kept_upper,
        constraints_used,
        unsupported_constraints,
    }))
}

#[derive(Debug, Default)]
struct MvfClipScreenTelemetry {
    target_eligible: bool,
    eligible_groups: usize,
    selected_groups: usize,
    capped_groups: usize,
    attempted_groups: usize,
    affine_groups: usize,
    missing_affine_groups: usize,
    clause_samples: usize,
    sample_capped_clauses: usize,
    clipped_clauses: usize,
    full_width_clauses: usize,
    empty_certificates: usize,
    unsupported_constraints: usize,
    deadline_hit: bool,
    admission_budget_ms: u128,
    reduction_elapsed_ms: u128,
    kept_ratios: Vec<f64>,
}

struct MvfClipDiagnosticState {
    selected_group_roots: BTreeSet<usize>,
    observed_group_roots: BTreeSet<usize>,
    max_samples: usize,
    outer_deadline: Option<Instant>,
    affine_budget: MvfAffineDiagnosticBudget,
    reduction_spent: Duration,
    telemetry: MvfClipScreenTelemetry,
}

impl MvfClipDiagnosticState {
    fn prepare(
        model_net: &BetaCrownModel,
        groups: &HashMap<Vec<u32>, (BoundedTensor, Vec<usize>)>,
        per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
        outer_deadline: Option<Instant>,
        target_eligible_override: Option<bool>,
    ) -> Self {
        let (admission_deadline, budget_ms) = phase_event_call_deadline(outer_deadline);
        let mut telemetry = MvfClipScreenTelemetry {
            admission_budget_ms: budget_ms,
            ..MvfClipScreenTelemetry::default()
        };
        telemetry.target_eligible = match target_eligible_override {
            Some(value) => value,
            None => match phase_event_target_eligible_before(
                model_net,
                groups,
                per_clause_input_bounds,
                Some(admission_deadline),
            ) {
                Ok(value) => value,
                Err(()) => {
                    telemetry.deadline_hit = true;
                    false
                }
            },
        };

        let mut eligible = Vec::new();
        if telemetry.target_eligible {
            for (bounds, active) in groups.values() {
                if Instant::now() >= admission_deadline {
                    // A partial HashMap traversal would make the capped sample
                    // iteration-order dependent. Decline the whole diagnostic
                    // instead; proof work is untouched.
                    telemetry.deadline_hit = true;
                    telemetry.target_eligible = false;
                    eligible.clear();
                    break;
                }
                if !active.is_empty() && single_centered_seed_axis(bounds).is_some() {
                    eligible.push(active[0]);
                }
            }
        }
        eligible.sort_unstable();
        eligible.dedup();
        telemetry.eligible_groups = eligible.len();
        let max_groups = mvf_clip_diag_max_groups();
        let selected_group_roots = eligible.iter().take(max_groups).copied().collect();
        telemetry.selected_groups = eligible.len().min(max_groups);
        telemetry.capped_groups = eligible.len().saturating_sub(max_groups);
        Self {
            selected_group_roots,
            observed_group_roots: BTreeSet::new(),
            max_samples: mvf_clip_diag_max_samples(),
            outer_deadline,
            affine_budget: MvfAffineDiagnosticBudget::new(
                outer_deadline,
                MVF_CLIP_DIAG_AFFINE_BUDGET,
            ),
            reduction_spent: Duration::ZERO,
            telemetry,
        }
    }

    fn selected_root(&self, node: &BoxNode) -> Option<usize> {
        (node.depth == 0)
            .then(|| node.active.first().copied())
            .flatten()
            .filter(|root| {
                self.selected_group_roots.contains(root)
                    && !self.observed_group_roots.contains(root)
            })
    }

    fn wants_affine_for_chunk(&self, wave: &[BoxNode], chunk: &[usize]) -> bool {
        self.telemetry.target_eligible
            && self.telemetry.clause_samples < self.max_samples
            && !self.affine_budget.is_exhausted()
            && chunk
                .iter()
                .any(|&index| self.selected_root(&wave[index]).is_some())
    }

    fn observe(
        &mut self,
        node: &BoxNode,
        clauses: &[Vec<OutputConstraint>],
        affine: Option<&MvfAffineEnclosure>,
    ) {
        let Some(root) = self.selected_root(node) else {
            return;
        };
        self.observed_group_roots.insert(root);
        self.telemetry.attempted_groups += 1;
        let Some(affine) = affine else {
            self.telemetry.missing_affine_groups += 1;
            return;
        };
        let [axis] = affine.seed_axes.as_slice() else {
            self.telemetry.missing_affine_groups += 1;
            return;
        };
        let (Some(domain_lower), Some(domain_upper)) = (
            node.bounds.lower().iter().nth(*axis).copied(),
            node.bounds.upper().iter().nth(*axis).copied(),
        ) else {
            self.telemetry.missing_affine_groups += 1;
            return;
        };
        let (domain_lower, domain_upper) = (
            f32_to_f64_exact(domain_lower),
            f32_to_f64_exact(domain_upper),
        );
        if !(domain_lower.is_finite() && domain_upper.is_finite() && domain_lower < domain_upper) {
            self.telemetry.missing_affine_groups += 1;
            return;
        }
        self.telemetry.affine_groups += 1;

        let reduction_started = Instant::now();
        let remaining_hard = MVF_CLIP_DIAG_REDUCTION_BUDGET.saturating_sub(self.reduction_spent);
        let reserve_budget = self
            .outer_deadline
            .map(|outer| {
                outer
                    .saturating_duration_since(reduction_started)
                    .checked_div(PHASE_EVENT_DIAG_OUTER_RESERVE_DIVISOR)
                    .unwrap_or(Duration::ZERO)
            })
            .unwrap_or(remaining_hard);
        let reduction_deadline = reduction_started
            .checked_add(remaining_hard.min(reserve_budget))
            .unwrap_or(reduction_started);

        for (offset, &clause_id) in node.active.iter().enumerate() {
            if self.telemetry.clause_samples >= self.max_samples {
                self.telemetry.sample_capped_clauses += node.active.len() - offset;
                break;
            }
            if Instant::now() >= reduction_deadline {
                self.telemetry.deadline_hit = true;
                self.telemetry.sample_capped_clauses += node.active.len() - offset;
                break;
            }
            let Some(clause) = clauses.get(clause_id) else {
                self.telemetry.unsupported_constraints += 1;
                continue;
            };
            let clip = match mvf_clip_clause_1d(
                affine,
                clause,
                domain_lower,
                domain_upper,
                Some(reduction_deadline),
            ) {
                Err(()) => {
                    self.telemetry.deadline_hit = true;
                    self.telemetry.sample_capped_clauses += node.active.len() - offset;
                    break;
                }
                Ok(Some(clip)) => clip,
                Ok(None) => {
                    self.telemetry.unsupported_constraints += clause.len();
                    continue;
                }
            };
            self.telemetry.clause_samples += 1;
            self.telemetry.unsupported_constraints += clip.unsupported_constraints;
            let ratio = clip.kept_ratio(domain_lower, domain_upper);
            self.telemetry.kept_ratios.push(ratio);
            if clip.empty() {
                self.telemetry.empty_certificates += 1;
            } else if ratio < 1.0 {
                self.telemetry.clipped_clauses += 1;
            } else {
                self.telemetry.full_width_clauses += 1;
            }
        }
        let elapsed = reduction_started.elapsed();
        self.reduction_spent = self.reduction_spent.saturating_add(elapsed);
        self.telemetry.reduction_elapsed_ms = self.reduction_spent.as_millis();
    }

    fn telemetry_line(&self) -> String {
        let mut ratios = self.telemetry.kept_ratios.clone();
        ratios.sort_by(f64::total_cmp);
        let quantile = |numerator: usize| -> f64 {
            if ratios.is_empty() {
                0.0
            } else {
                ratios[(ratios.len() - 1).saturating_mul(numerator) / 100]
            }
        };
        format!(
            "{NN4SYS_MVF_CLIP_TELEMETRY_MARKER} target_eligible={} \
eligible_groups={} selected_groups={} cap_skipped_groups={} attempted_groups={} \
affine_groups={} missing_affine_groups={} clause_samples={} sample_capped_clauses={} \
clipped_clauses={} full_width_clauses={} empty_certificates={} \
unsupported_constraints={} kept_ratio_p00={:.9} kept_ratio_p50={:.9} \
kept_ratio_p95={:.9} kept_ratio_p100={:.9} deadline_hit={} \
admission_budget_ms={} reduction_elapsed_ms={} verdict_authority=false \
proof_state_mutations=0 clause_tails_merged=false",
            self.telemetry.target_eligible,
            self.telemetry.eligible_groups,
            self.telemetry.selected_groups,
            self.telemetry.capped_groups,
            self.telemetry.attempted_groups,
            self.telemetry.affine_groups,
            self.telemetry.missing_affine_groups,
            self.telemetry.clause_samples,
            self.telemetry.sample_capped_clauses,
            self.telemetry.clipped_clauses,
            self.telemetry.full_width_clauses,
            self.telemetry.empty_certificates,
            self.telemetry.unsupported_constraints,
            ratios.first().copied().unwrap_or(0.0),
            quantile(50),
            quantile(95),
            ratios.last().copied().unwrap_or(0.0),
            self.telemetry.deadline_hit,
            self.telemetry.admission_budget_ms,
            self.telemetry.reduction_elapsed_ms,
        )
    }
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
/// Malformed bounds and non-finite constants fail closed before comparison.
fn interval64_is_finite_and_ordered(output: &Interval64) -> bool {
    !output.lower.is_empty()
        && output.lower.shape() == output.upper.shape()
        && output
            .lower
            .iter()
            .zip(&output.upper)
            .all(|(&lower, &upper)| lower.is_finite() && upper.is_finite() && lower <= upper)
}

fn constraint_provably_false_f64(output: &Interval64, c: &OutputConstraint) -> bool {
    if !interval64_is_finite_and_ordered(output) {
        return false;
    }
    let l = |i: usize| output.lower.iter().nth(i).copied();
    let u = |i: usize| output.upper.iter().nth(i).copied();
    match c {
        OutputConstraint::LessEqConst(i, k) => k.is_finite() && l(*i).is_some_and(|v| v > *k),
        OutputConstraint::LessThanConst(i, k) => k.is_finite() && l(*i).is_some_and(|v| v >= *k),
        OutputConstraint::GreaterEqConst(i, k) => k.is_finite() && u(*i).is_some_and(|v| v < *k),
        OutputConstraint::GreaterThanConst(i, k) => k.is_finite() && u(*i).is_some_and(|v| v <= *k),
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
pub(super) fn clause_provably_unsat_f64(output: &Interval64, clause: &[OutputConstraint]) -> bool {
    !clause.is_empty()
        && interval64_is_finite_and_ordered(output)
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
            mid(*i).is_some_and(|m| f32_to_f64_exact(m) > *k)
        }
        OutputConstraint::GreaterEqConst(i, k) | OutputConstraint::GreaterThanConst(i, k) => {
            mid(*i).is_some_and(|m| f32_to_f64_exact(m) < *k)
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
fn merge_verified_flags(mut verified: Vec<bool>, additional: Option<Vec<bool>>) -> Vec<bool> {
    if let Some(additional) = additional.filter(|additional| additional.len() == verified.len()) {
        for (verified, additional) in verified.iter_mut().zip(additional) {
            *verified |= additional;
        }
    }
    verified
}

pub(super) fn refine_clause_boxes(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Vec<bool> {
    // Exact one-axis and few-axis complete covers. Each may return only
    // independently retired group certificates after its bounded slice.
    // Seed their monotone union into the mature screen so no certified clause
    // is recomputed, then defensively OR the same vector into the result. A
    // false seed always means unproven and proceeds unchanged downstream.
    let nd_verified = super::nn4sys_nd_cover::try_nn4sys_nd_complete_cover(
        model_net,
        input,
        clauses,
        per_clause_input_bounds,
        deadline,
    );
    let scalar_verified = super::nn4sys_scalar_cover::try_nn4sys_scalar_complete_cover(
        model_net,
        input,
        clauses,
        per_clause_input_bounds,
        deadline,
    );

    let mut independently_verified: Option<Vec<bool>> = None;
    for candidate in [nd_verified, scalar_verified] {
        let Some(candidate) = candidate.filter(|candidate| candidate.len() == clauses.len()) else {
            continue;
        };
        if let Some(accumulated) = independently_verified.as_mut() {
            for (accumulated, candidate) in accumulated.iter_mut().zip(candidate) {
                *accumulated |= candidate;
            }
        } else {
            independently_verified = Some(candidate);
        }
    }
    if let Some(verified) = independently_verified
        .as_ref()
        .filter(|verified| verified.iter().all(|&value| value))
    {
        return verified.clone();
    }
    let independent_seed = independently_verified
        .as_deref()
        .filter(|verified| verified.len() == clauses.len());
    let (verified, _) = refine_clause_boxes_counted_seeded(
        model_net,
        input,
        clauses,
        per_clause_input_bounds,
        gemm_engine,
        deadline,
        independent_seed,
        None,
    );
    merge_verified_flags(verified, independently_verified)
}

/// [`refine_clause_boxes`] plus the number of box nodes processed — the
/// budget-shape assertions of the plateau-clause tests key on it.
#[cfg(test)]
fn refine_clause_boxes_counted(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> (Vec<bool>, usize) {
    refine_clause_boxes_counted_with_phase_target_override(
        model_net,
        input,
        clauses,
        per_clause_input_bounds,
        gemm_engine,
        deadline,
        None,
    )
}

#[cfg(test)]
fn refine_clause_boxes_counted_with_phase_target_override(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    phase_target_override: Option<bool>,
) -> (Vec<bool>, usize) {
    refine_clause_boxes_counted_seeded(
        model_net,
        input,
        clauses,
        per_clause_input_bounds,
        gemm_engine,
        deadline,
        None,
        phase_target_override,
    )
}

/// Run the mature box-refinement screen with independently authenticated
/// per-clause proof facts already discharged. A cardinality mismatch discards
/// the seed wholesale. Seeded clauses never enter a root group, so the legacy
/// deadline is spent only on obligations that remain open; the final result
/// still derives from the same `failed/open_nodes` invariant.
#[allow(clippy::too_many_arguments)]
fn refine_clause_boxes_counted_seeded(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    initially_verified: Option<&[bool]>,
    phase_target_override: Option<bool>,
) -> (Vec<bool>, usize) {
    let start = Instant::now();
    let n = clauses.len();
    let initially_verified = initially_verified.filter(|verified| verified.len() == n);
    let seeded_clauses = initially_verified
        .map(|verified| verified.iter().filter(|&&value| value).count())
        .unwrap_or(0);
    // Sound f64 leaf escalation (Graph models with a fully-supported op set
    // only; NY_F64_LEAF=0 disables). None => leaves fail exactly as before.
    let f64_leaf = f64_leaf_escalation(model_net);
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
    let mut groups: ClauseBoxGroups = HashMap::new();
    for (idx, clause) in clauses.iter().enumerate() {
        if initially_verified.is_some_and(|verified| verified[idx]) {
            continue;
        }
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
    // Exact production gate for the finite-deadline pilot schedule. A gate
    // error (deadline during structural admission) declines to the legacy
    // schedule; it never grants proof authority or broadens model coverage.
    let exact_mscn_2048_dual =
        exact_mscn_2048_dual_screen_before(model_net, &groups, per_clause_input_bounds, deadline)
            .unwrap_or(false);
    let mvf_deadline_pilot_active = exact_mscn_2048_dual
        && deadline.is_some()
        && f64_leaf.as_ref().is_some_and(|esc| esc.batch && esc.mvf);
    let effective_mvf_wave_cap = if mvf_deadline_pilot_active {
        wave_size().min(mvf_deadline_wave_size())
    } else {
        0
    };
    // The exact auto-profile has 41 few-axis clauses. Only after all 41 have
    // independently retired may its measured closer pre-shed optional CROWN;
    // a partial exact-cover run keeps the mature adaptive policy unchanged.
    let auto_128_240_seeded_closer = seeded_clauses == 41
        && super::nn4sys_nd_cover::authentic_128_240_auto_profile(
            model_net,
            input,
            clauses,
            per_clause_input_bounds,
        );
    // Adaptive per-run CROWN cost gate (NY_SCREEN_CROWN_MS tunes generic
    // screens). The active finite-deadline 2048-wide dual batch+MVF pilot
    // pre-sheds its measured-useless first probe so capped Rayon scheduling
    // cannot choose a random winner. Explicit `NY_SCREEN_CROWN_MS=0` still
    // disables the gate and therefore preserves the requested CROWN attempts.
    let mut crown_gate = CrownCostGate::from_env(f64_leaf.as_ref().is_some_and(|esc| esc.batch));
    if mvf_deadline_pilot_active
        || auto_128_240_seeded_closer
        || (seeded_clauses > 0 && seeded_crown_preshed_enabled())
    {
        crown_gate.force_shed();
    }
    // Dark diagnostic only: point-JVP events are counted and discarded.
    // Nothing in the verifier reads this telemetry, and the root groups are
    // borrowed immutably. Gate OFF is the legacy path with no probe walks.
    if phase_event_diag_enabled() {
        let (phase_deadline, budget_ms) = phase_event_call_deadline(deadline);
        let phase = collect_phase_event_screen_telemetry(
            model_net,
            &groups,
            per_clause_input_bounds,
            phase_event_diag_max_groups(),
            phase_deadline,
            budget_ms,
            phase_target_override,
        );
        if let Some(line) = phase_event_telemetry_line_if(true, &phase) {
            eprintln!("{line}");
        }
    }
    // Independent dark MVF-Clip diagnostic. Admission reuses the exact
    // MSCN-dual graph/property gate above. The state owns only counters and a
    // deterministic root-group sample; it never reaches queue construction.
    let mut mvf_clip_diag = mvf_clip_diag_enabled().then(|| {
        MvfClipDiagnosticState::prepare(
            model_net,
            &groups,
            per_clause_input_bounds,
            deadline,
            phase_target_override,
        )
    });
    let mut queue = ordered_root_queue(groups);

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

        // A finite-deadline batched MVF screen must adjudicate a small wave
        // before starting more derivative walks.  Otherwise a large root wave
        // can consume the entire slice in precomputation and have every result
        // discarded by `bound_node`'s deadline guard.
        let wave_len = scheduled_wave_len(
            queue.len(),
            f64_leaf.as_ref(),
            deadline,
            exact_mscn_2048_dual,
        );
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
        // weights. Sound and per-box-isolated by the walks' contracts. A
        // non-deadline chunk failure leaves those boxes on the byte-identical
        // per-box lane; deadline expiry stops the whole wave conservatively
        // and never launches that expensive fallback.
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
        let mut batched_deadline_stop = false;
        if let Some(esc) = f64_leaf.as_ref().filter(|e| e.batch && wave.len() >= 2) {
            // Weight preparation is a one-time parallel transpose/absolute
            // cache build. Individual weight transforms are currently atomic,
            // so fence it on both sides: never start after expiry and never
            // publish a late cache build into a graph walk. The exact dual
            // model's observed preparation is bounded well below one pilot
            // wave; cooperative graph polling begins in the calls below.
            let weights = if deadline.is_some_and(|d| Instant::now() >= d) {
                batched_deadline_stop = true;
                None
            } else {
                Some(
                    esc.weight_cache
                        .get_or_init(|| esc.graph.build_f64_weight_cache()),
                )
            };
            if deadline.is_some_and(|d| Instant::now() >= d) {
                batched_deadline_stop = true;
            }
            if let Some(weights) = weights.filter(|_| !batched_deadline_stop) {
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
                        batched_deadline_stop = true;
                        break;
                    }
                    let inputs: Vec<Interval64> = chunk
                        .iter()
                        .map(|&i| {
                            Interval64::from_f32(wave[i].bounds.lower(), wave[i].bounds.upper())
                        })
                        .collect();
                    match esc.graph.propagate_ibp_f64_cells_cached_with_deadline(
                        &inputs,
                        Some(weights),
                        deadline,
                    ) {
                        Ok(outs) => {
                            esc.batched_boxes.fetch_add(outs.len(), Ordering::Relaxed);
                            for (&i, out) in chunk.iter().zip(outs) {
                                cell_outs[i] = Some(out);
                            }
                        }
                        Err(error) if error.is_deadline_exceeded() => {
                            batched_deadline_stop = true;
                            break;
                        }
                        Err(_) => {
                            esc.batched_fallbacks.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                if !batched_deadline_stop {
                    for chunk in mvf_idx.chunks(mvf_chunk()) {
                        if deadline.is_some_and(|d| Instant::now() >= d) {
                            batched_deadline_stop = true;
                            break;
                        }
                        let boxes: Vec<Interval64> = chunk
                            .iter()
                            .map(|&i| {
                                Interval64::from_f32(wave[i].bounds.lower(), wave[i].bounds.upper())
                            })
                            .collect();
                        let wants_affine = mvf_clip_diag
                            .as_ref()
                            .is_some_and(|diag| diag.wants_affine_for_chunk(&wave, chunk));
                        let result = if wants_affine {
                            let affine_budget = &mut mvf_clip_diag
                                .as_mut()
                                .expect("affine request requires diagnostic state")
                                .affine_budget;
                            esc.graph.propagate_ibp_f64_centered_mono_affine_diagnostic_cells_cached_with_deadline(
                            &boxes,
                            esc.mono,
                            Some(weights),
                            affine_budget,
                            deadline,
                        )
                        } else {
                            esc.graph
                                .propagate_ibp_f64_centered_mono_cells_cached_with_deadline(
                                    &boxes,
                                    esc.mono,
                                    Some(weights),
                                    deadline,
                                )
                        };
                        match result {
                            Ok(outs) => {
                                esc.batched_boxes.fetch_add(outs.len(), Ordering::Relaxed);
                                esc.batched_mvf_boxes
                                    .fetch_add(outs.len(), Ordering::Relaxed);
                                for (&i, out) in chunk.iter().zip(outs) {
                                    if let Some(diag) = mvf_clip_diag.as_mut() {
                                        diag.observe(
                                            &wave[i],
                                            clauses,
                                            out.affine_diagnostic.as_ref(),
                                        );
                                    }
                                    if esc.mono {
                                        record_mono_stats(esc, &out);
                                    }
                                    cell_outs[i] = Some(out.value);
                                    centered_outs[i] = Some(out.centered);
                                    mono_outs[i] = out.mono;
                                }
                            }
                            Err(error) if error.is_deadline_exceeded() => {
                                batched_deadline_stop = true;
                                break;
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
            }
        }

        if batched_deadline_stop {
            // Preserve every root obligation for the conservative deadline
            // tail.  In particular, do NOT retry a cooperatively-cancelled
            // batch through the per-node lane: that repeats the expensive
            // graph walk after the proof slice has already expired.
            deadline_hit = true;
            queue.extend(wave);
            break;
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

        // Do not publish a partial/scheduler-dependent wave transition after
        // the deadline. Requeue every root obligation and let the common
        // conservative deadline tail mark it unproven.
        if completed_wave_must_be_discarded(deadline, Instant::now()) {
            deadline_hit = true;
            queue.extend(wave);
            break;
        }

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
        seeded_clauses,
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
        exact_mscn_2048_dual,
        finite_deadline_mvf_pilot_active = mvf_deadline_pilot_active,
        auto_128_240_seeded_closer,
        finite_deadline_mvf_wave_cap = effective_mvf_wave_cap,
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
    if let Some(diag) = mvf_clip_diag {
        eprintln!("{}", diag.telemetry_line());
    }
    (result, nodes_processed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_crown_preshed_requires_exact_opt_in() {
        assert!(seeded_crown_preshed_enabled_from_value(Some("1")));
        for value in [None, Some(""), Some("0"), Some("true"), Some("01")] {
            assert!(!seeded_crown_preshed_enabled_from_value(value));
        }
    }

    #[test]
    fn independently_verified_clause_flags_merge_monotonically_and_fail_closed_on_mismatch() {
        assert_eq!(
            merge_verified_flags(vec![false, true, false], Some(vec![true, false, false])),
            vec![true, true, false]
        );
        assert_eq!(
            merge_verified_flags(vec![false, true], Some(vec![true])),
            vec![false, true],
            "a cardinality mismatch must discard the additional lane"
        );
    }

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

    fn interval64(lo: Vec<f64>, hi: Vec<f64>) -> Interval64 {
        Interval64 {
            lower: ArrayD::from_shape_vec(vec![lo.len()], lo).unwrap(),
            upper: ArrayD::from_shape_vec(vec![hi.len()], hi).unwrap(),
        }
    }

    #[test]
    fn f64_clause_proof_rejects_nonfinite_inverted_or_mismatched_boxes() {
        let clause = [OutputConstraint::LessEqConst(0, 0.0)];
        for output in [
            interval64(vec![f64::INFINITY], vec![f64::INFINITY]),
            interval64(vec![2.0], vec![1.0]),
            interval64(vec![1.0, 2.0], vec![2.0]),
        ] {
            assert!(!clause_provably_unsat_f64(&output, &clause));
        }

        let finite = interval64(vec![1.0], vec![2.0]);
        assert!(!clause_provably_unsat_f64(
            &finite,
            &[OutputConstraint::LessEqConst(0, f64::NAN)]
        ));
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

    #[test]
    fn root_queue_order_is_deterministic_by_box_bits() {
        let boxes = [
            (boxed(vec![0.0], vec![0.75]), 7usize),
            (boxed(vec![0.0], vec![0.25]), 2usize),
            (boxed(vec![0.0], vec![0.5]), 5usize),
        ];
        let mut groups = ClauseBoxGroups::new();
        for (bounds, clause) in boxes {
            groups.insert(
                box_key(&bounds).expect("contiguous box"),
                (bounds, vec![clause]),
            );
        }

        let queue = ordered_root_queue(groups);
        assert_eq!(
            queue.iter().map(|node| node.active[0]).collect::<Vec<_>>(),
            vec![2, 5, 7],
            "root stack must follow ascending bit-key order; tail pops are stable"
        );
    }

    #[test]
    fn completed_wave_is_discarded_as_one_unit_after_deadline() {
        let now = Instant::now();
        assert!(completed_wave_must_be_discarded(Some(now), now));
        assert!(completed_wave_must_be_discarded(
            Some(now),
            now + Duration::from_nanos(1)
        ));
        assert!(!completed_wave_must_be_discarded(
            Some(now + Duration::from_secs(1)),
            now
        ));
        assert!(!completed_wave_must_be_discarded(None, now));
    }

    /// y = (1/3)·x + 0.7 as a 1-node Graph model. 1/3 and 0.7 have no exact
    /// f32 representation, so the sound f32 forward carries a rounding floor
    /// of ~1 f32 ULP (~6e-8 at |y|≈0.8) — wider than the 1e-9 band margins
    /// these tests use.
    fn build_band_graph() -> GraphNetwork {
        let w = ndarray::arr2(&[[1.0f32 / 3.0]]);
        let b = ndarray::arr1(&[0.7f32]);
        let linear = ny_propagate::layers::LinearLayer::new(w, Some(b)).unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(ny_propagate::GraphNode::from_input(
            "lin",
            Layer::Linear(linear),
        ));
        graph.set_output("lin");
        graph
    }

    /// z = 2x - 1 -> ReLU(z), with one analytic phase event at x=0.5.
    fn build_phase_diag_graph() -> GraphNetwork {
        let linear = ny_propagate::layers::LinearLayer::new(
            ndarray::arr2(&[[2.0f32]]),
            Some(ndarray::arr1(&[-1.0f32])),
        )
        .unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(ny_propagate::GraphNode::from_input(
            "lin",
            Layer::Linear(linear),
        ));
        graph.add_node(ny_propagate::GraphNode::new(
            "relu",
            Layer::ReLU(ny_propagate::layers::ReLULayer),
            vec!["lin".to_string()],
        ));
        graph.set_output("relu");
        graph
    }

    #[test]
    fn independently_verified_clauses_seed_legacy_state_and_mismatch_fails_closed() {
        let input = boxed(vec![0.0], vec![1.0]);
        let clauses = vec![
            vec![OutputConstraint::LessEqConst(0, -1.0)],
            vec![OutputConstraint::LessEqConst(0, -1.0)],
        ];
        let clause_box = std::collections::BTreeMap::from([(0usize, (0.0, 1.0))]);
        let boxes = vec![clause_box.clone(), clause_box];

        let seeded_model = BetaCrownModel::Graph(Box::new(build_phase_diag_graph()));
        let seeded = refine_clause_boxes_counted_seeded(
            &seeded_model,
            &input,
            &clauses,
            &boxes,
            None,
            Some(Instant::now()),
            Some(&[true, false]),
            None,
        );
        assert_eq!(seeded, (vec![true, false], 0));

        let mismatched_model = BetaCrownModel::Graph(Box::new(build_phase_diag_graph()));
        let mismatched = refine_clause_boxes_counted_seeded(
            &mismatched_model,
            &input,
            &clauses,
            &boxes,
            None,
            Some(Instant::now()),
            Some(&[true]),
            None,
        );
        assert_eq!(
            mismatched,
            (vec![false, false], 0),
            "a cardinality mismatch must seed no proof facts"
        );
    }

    #[test]
    fn phase_event_env_parsers_are_exact_and_bounded() {
        assert!(!phase_event_diag_enabled_from_value(None));
        assert!(phase_event_diag_enabled_from_value(Some("1")));
        for malformed in ["", "0", "00", "true", " 1", "1 "] {
            assert!(!phase_event_diag_enabled_from_value(Some(malformed)));
        }

        assert_eq!(
            phase_event_diag_max_groups_from_value(None),
            PHASE_EVENT_DIAG_MAX_GROUPS
        );
        assert_eq!(phase_event_diag_max_groups_from_value(Some("1")), 1);
        assert_eq!(phase_event_diag_max_groups_from_value(Some("00256")), 256);
        assert_eq!(
            phase_event_diag_max_groups_from_value(Some("4096")),
            PHASE_EVENT_DIAG_HARD_MAX_GROUPS
        );
        for malformed in ["", "0", "4097", "-1", "+1", " 1", "1 "] {
            assert_eq!(
                phase_event_diag_max_groups_from_value(Some(malformed)),
                PHASE_EVENT_DIAG_MAX_GROUPS
            );
        }
    }

    #[test]
    fn mvf_clip_env_parsers_are_exact_default_off_and_bounded() {
        assert!(!mvf_clip_diag_enabled_from_value(None));
        assert!(mvf_clip_diag_enabled_from_value(Some("1")));
        for disabled_or_malformed in ["", "0", "00", "true", " 1", "1 ", "+1"] {
            assert!(!mvf_clip_diag_enabled_from_value(Some(
                disabled_or_malformed
            )));
        }

        assert_eq!(
            mvf_clip_diag_max_groups_from_value(None),
            MVF_CLIP_DIAG_MAX_GROUPS
        );
        assert_eq!(mvf_clip_diag_max_groups_from_value(Some("1")), 1);
        assert_eq!(
            mvf_clip_diag_max_groups_from_value(Some("4096")),
            MVF_CLIP_DIAG_HARD_MAX_GROUPS
        );
        for malformed in ["", "0", "4097", "-1", "+1", " 1", "1 "] {
            assert_eq!(
                mvf_clip_diag_max_groups_from_value(Some(malformed)),
                MVF_CLIP_DIAG_MAX_GROUPS
            );
        }

        assert_eq!(
            mvf_clip_diag_max_samples_from_value(None),
            MVF_CLIP_DIAG_MAX_SAMPLES
        );
        assert_eq!(mvf_clip_diag_max_samples_from_value(Some("1")), 1);
        assert_eq!(
            mvf_clip_diag_max_samples_from_value(Some("16384")),
            MVF_CLIP_DIAG_HARD_MAX_SAMPLES
        );
        for malformed in ["", "0", "16385", "-1", "+1", " 1", "1 "] {
            assert_eq!(
                mvf_clip_diag_max_samples_from_value(Some(malformed)),
                MVF_CLIP_DIAG_MAX_SAMPLES
            );
        }
    }

    fn exact_opposing_tail_affine() -> MvfAffineEnclosure {
        MvfAffineEnclosure {
            seed_axes: vec![0],
            centers: vec![0.0],
            // y0=x, y1=-x exactly.
            coefficients: vec![
                ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, -1.0]).unwrap(),
            ],
            remainder: Interval64 {
                lower: ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
                upper: ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
            },
        }
    }

    #[test]
    fn mvf_clip_keeps_opposing_clause_tails_isolated_and_intersects_within_clause() {
        let affine = exact_opposing_tail_affine();
        let left_clause = vec![OutputConstraint::LessEqConst(0, -0.25)];
        let right_clause = vec![OutputConstraint::GreaterEqConst(0, 0.25)];
        let left = mvf_clip_clause_1d(&affine, &left_clause, -1.0, 1.0, None)
            .unwrap()
            .unwrap();
        let right = mvf_clip_clause_1d(&affine, &right_clause, -1.0, 1.0, None)
            .unwrap()
            .unwrap();
        assert!(left.kept_upper < 0.0, "left tail must stay left: {left:?}");
        assert!(
            right.kept_lower > 0.0,
            "right tail must stay right: {right:?}"
        );
        assert!(
            left.kept_upper < right.kept_lower,
            "separate clauses must not union opposing tails"
        );

        let impossible_conjunction = vec![
            OutputConstraint::LessEqConst(0, -0.25),
            OutputConstraint::GreaterEqConst(0, 0.25),
        ];
        let empty = mvf_clip_clause_1d(&affine, &impossible_conjunction, -1.0, 1.0, None)
            .unwrap()
            .unwrap();
        assert!(
            empty.empty(),
            "conjunctive row clips must intersect to empty"
        );
        assert_eq!(empty.kept_ratio(-1.0, 1.0), 0.0);
    }

    #[test]
    fn mvf_clip_clause_checks_the_deadline_before_each_constraint() {
        let affine = exact_opposing_tail_affine();
        let clause = vec![OutputConstraint::LessEqConst(0, 0.5)];
        assert!(matches!(
            mvf_clip_clause_1d(&affine, &clause, -1.0, 1.0, Some(Instant::now())),
            Err(())
        ));
    }

    #[test]
    fn mvf_clip_random_samples_that_truly_satisfy_clause_are_always_retained() {
        let affine = exact_opposing_tail_affine();
        let clauses = [
            vec![
                OutputConstraint::GreaterEqConst(0, -0.7),
                OutputConstraint::LessEqConst(0, 0.3),
            ],
            vec![OutputConstraint::LessEq(0, 1)],    // x <= -x
            vec![OutputConstraint::GreaterEq(0, 1)], // x >= -x
            vec![
                OutputConstraint::GreaterThanConst(0, -0.2),
                OutputConstraint::LessThanConst(0, 0.8),
            ],
        ];
        let clips: Vec<MvfClauseClip1d> = clauses
            .iter()
            .map(|clause| {
                mvf_clip_clause_1d(&affine, clause, -1.0, 1.0, None)
                    .unwrap()
                    .unwrap()
            })
            .collect();

        let mut state = 0xC11F_5EED_D1A6_0001u64;
        for _ in 0..10_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            let x = -1.0 + 2.0 * unit;
            let ys = [x, -x];
            for (clause, clip) in clauses.iter().zip(&clips) {
                let truly_satisfies = clause.iter().all(|constraint| match constraint {
                    OutputConstraint::LessEqConst(i, k) => ys[*i] <= *k,
                    OutputConstraint::LessThanConst(i, k) => ys[*i] < *k,
                    OutputConstraint::GreaterEqConst(i, k) => ys[*i] >= *k,
                    OutputConstraint::GreaterThanConst(i, k) => ys[*i] > *k,
                    OutputConstraint::LessEq(i, j) => ys[*i] <= ys[*j],
                    OutputConstraint::LessThan(i, j) => ys[*i] < ys[*j],
                    OutputConstraint::GreaterEq(i, j) => ys[*i] >= ys[*j],
                    OutputConstraint::GreaterThan(i, j) => ys[*i] > ys[*j],
                    _ => false,
                });
                if truly_satisfies {
                    assert!(
                        clip.kept_lower <= x && x <= clip.kept_upper,
                        "true clause point x={x} escaped retained interval {clip:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn mvf_clip_telemetry_marker_is_stable_and_denies_verdict_authority() {
        let state = MvfClipDiagnosticState {
            selected_group_roots: BTreeSet::new(),
            observed_group_roots: BTreeSet::new(),
            max_samples: 1,
            outer_deadline: None,
            affine_budget: MvfAffineDiagnosticBudget::new(None, MVF_CLIP_DIAG_AFFINE_BUDGET),
            reduction_spent: Duration::ZERO,
            telemetry: MvfClipScreenTelemetry {
                target_eligible: true,
                eligible_groups: 2,
                selected_groups: 1,
                clause_samples: 1,
                empty_certificates: 1,
                kept_ratios: vec![0.0],
                ..MvfClipScreenTelemetry::default()
            },
        };
        let line = state.telemetry_line();
        assert!(
            line.starts_with("NY_NN4SYS_MVF_CLIP_DIAG_V1 target_eligible=true eligible_groups=2")
        );
        assert!(line.contains("empty_certificates=1"));
        assert!(line.contains("verdict_authority=false"));
        assert!(line.contains("proof_state_mutations=0"));
        assert!(line.contains("clause_tails_merged=false"));
    }

    #[test]
    fn phase_event_stderr_marker_is_default_off_and_exact_when_enabled() {
        let phase = PhaseEventScreenTelemetry {
            target_eligible: true,
            property_census_complete: true,
            property_clauses: 1,
            property_dim_0: 2,
            property_dim_1: 3,
            property_dim_2: 4,
            property_dim_3: 5,
            property_dim_4: 6,
            property_dim_5: 7,
            property_dim_overflow: 8,
            eligible_groups: 9,
            probed_groups: 10,
            capped_groups: 11,
            deadline_capped_groups: 12,
            declined_groups: 13,
            collector_errors: 14,
            deadline_errors: 15,
            deadline_hit: true,
            relu_nodes: 16,
            preactivations: 17,
            candidates: 18,
            at_point_kinks: 19,
            stationary: 20,
            slope_straddles_zero: 21,
            non_finite: 22,
            outside_scan: 23,
            candidates_p50: 24,
            candidates_p95: 25,
            candidates_max: 26,
            budget_ms: 27,
            elapsed_ms: 28,
        };
        assert_eq!(phase_event_telemetry_line_if(false, &phase), None);
        assert_eq!(
            phase_event_telemetry_line_if(true, &phase).as_deref(),
            Some(
                "NY_NN4SYS_1D_PHASE_EVENTS_V1 target_eligible=true \
property_census_complete=true property_clauses=1 property_dim_0=2 \
property_dim_1=3 property_dim_2=4 property_dim_3=5 property_dim_4=6 \
property_dim_5=7 property_dim_overflow=8 eligible_groups=9 attempted_groups=10 \
cap_skipped_groups=11 deadline_skipped_groups=12 declined_groups=13 \
collector_errors=14 deadline_errors=15 deadline_hit=true relu_nodes=16 \
preactivations=17 candidates=18 at_point_kinks=19 stationary=20 \
slope_straddles_zero=21 non_finite=22 outside_scan=23 candidates_p50=24 \
candidates_p95=25 candidates_max=26 budget_ms=27 elapsed_ms=28 \
verdict_authority=false"
            )
        );
    }

    #[test]
    fn phase_event_target_gate_rejects_generic_graph_and_partial_clause_box() {
        let graph = build_phase_diag_graph();
        assert!(!is_nn4sys_mscn_dual_graph(&graph));

        let mut full = std::collections::BTreeMap::new();
        for axis in 0..22 * 14 {
            full.insert(axis, (0.0, 0.0));
        }
        assert!(has_exact_mscn_dual_clause_surface(&full));
        full.remove(&17);
        assert!(!has_exact_mscn_dual_clause_surface(&full));
    }

    #[test]
    fn phase_event_structure_prepass_is_size_and_deadline_bounded() {
        let mut oversized = GraphNetwork::new();
        for ordinal in 0..=MSCN_DUAL_MAX_GRAPH_NODES {
            oversized.add_node(ny_propagate::GraphNode::from_input(
                format!("relu_{ordinal}"),
                Layer::ReLU(ny_propagate::layers::ReLULayer),
            ));
        }
        oversized.set_output(format!("relu_{MSCN_DUAL_MAX_GRAPH_NODES}"));
        assert_eq!(oversized.num_nodes(), MSCN_DUAL_MAX_GRAPH_NODES + 1);
        assert_eq!(
            mscn_dual_structure_before(&oversized, None),
            Ok(None),
            "public node count must reject before topological admission"
        );

        assert_eq!(
            mscn_dual_structure_before(&build_phase_diag_graph(), Some(Instant::now())),
            Err(()),
            "expired structural prepass must stop immediately"
        );
    }

    #[test]
    #[cfg(feature = "external-vnncomp")]
    fn phase_event_gate_loads_both_real_dual_models_and_rejects_edge_collision() {
        let dir = std::env::var("NY_TEST_NN4SYS_ONNX_DIR")
            .expect("external-vnncomp MSCN dual conformance requires NY_TEST_NN4SYS_ONNX_DIR");
        for (file, expected) in [
            ("mscn_128d_dual.onnx", MscnDualVariant::D128),
            ("mscn_2048d_dual.onnx", MscnDualVariant::D2048),
        ] {
            let path = std::path::Path::new(&dir).join(file);
            let graph = crate::commands::vnncomp::load_graph_network(&path)
                .unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
            let structure = mscn_dual_structure(&graph).expect("bounded supported structure");
            let digest = mscn_dual_structure_digest(&structure);
            eprintln!("{file} structural digest: {digest}");
            assert_eq!(
                classify_mscn_dual_digest(&digest),
                Some(expected),
                "{file} exact structural allowlist"
            );

            let mut near_collision = structure.clone();
            let changed = near_collision.nodes.iter_mut().find_map(|node| {
                node.inputs.iter_mut().find_map(|input| match input {
                    StructuralInput::Node(parent) if *parent > 0 => {
                        *parent -= 1;
                        Some(())
                    }
                    _ => None,
                })
            });
            assert!(
                changed.is_some(),
                "fixture must contain a rewritable non-root edge"
            );
            assert!(
                structure
                    .nodes
                    .iter()
                    .zip(&near_collision.nodes)
                    .all(|(original, collision)| original.layer == collision.layer
                        && original.output_shape == collision.output_shape),
                "near-collision must preserve the exact layer/shape histogram"
            );
            let collision_digest = mscn_dual_structure_digest(&near_collision);
            assert_ne!(collision_digest, digest);
            assert_eq!(
                classify_mscn_dual_digest(&collision_digest),
                None,
                "same layers/shapes with one changed edge must fail closed"
            );

            let model = BetaCrownModel::Graph(Box::new(graph));
            let mut upper = vec![1.0; 22 * 14];
            upper[0] = 1.1;
            let root = BoundedTensor::new(
                ArrayD::from_shape_vec(ndarray::IxDyn(&[22, 14]), vec![1.0; 22 * 14]).unwrap(),
                ArrayD::from_shape_vec(ndarray::IxDyn(&[22, 14]), upper).unwrap(),
            )
            .unwrap();
            let mut groups = HashMap::new();
            groups.insert(box_key(&root).unwrap(), (root, vec![0]));
            let mut clause_box = std::collections::BTreeMap::new();
            for axis in 0..22 * 14 {
                clause_box.insert(axis, (1.0, if axis == 0 { 1.1 } else { 1.0 }));
            }
            assert!(phase_event_target_eligible(
                &model,
                &groups,
                std::slice::from_ref(&clause_box)
            ));
            assert_eq!(
                exact_mscn_2048_dual_screen_before(
                    &model,
                    &groups,
                    std::slice::from_ref(&clause_box),
                    None,
                ),
                Ok(expected == MscnDualVariant::D2048),
                "production pilot gate must admit only the real 2048-wide dual"
            );
            assert!(
                !phase_event_target_eligible(
                    &model,
                    &groups,
                    &[clause_box.clone(), clause_box.clone()]
                ),
                "non-official property cardinality must fail closed"
            );
            if expected == MscnDualVariant::D128 {
                let telemetry = collect_phase_event_screen_telemetry(
                    &model,
                    &groups,
                    &[clause_box],
                    1,
                    Instant::now() + Duration::from_secs(5),
                    5_000,
                    None,
                );
                assert!(telemetry.target_eligible);
                assert_eq!(
                    telemetry.probed_groups, 1,
                    "real eligible gate must execute one point-JVP"
                );
                assert_eq!(telemetry.collector_errors, 0);
                assert!(!telemetry.deadline_hit);
            }
        }
    }

    #[test]
    fn phase_event_dark_probe_censuses_and_collects_without_authority() {
        let graph = build_phase_diag_graph();
        let model = BetaCrownModel::Graph(Box::new(graph));
        let root = boxed(vec![0.0], vec![1.0]);
        let mut groups = HashMap::new();
        groups.insert(box_key(&root).unwrap(), (root, vec![0]));
        let mut clause_box = std::collections::BTreeMap::new();
        clause_box.insert(0usize, (0.0, 1.0));

        let telemetry = collect_phase_event_screen_telemetry(
            &model,
            &groups,
            &[clause_box],
            8,
            Instant::now() + Duration::from_secs(1),
            1_000,
            Some(true),
        );
        assert_eq!(telemetry.property_clauses, 1);
        assert_eq!(telemetry.property_dim_1, 1);
        assert_eq!(telemetry.eligible_groups, 1);
        assert_eq!(telemetry.probed_groups, 1);
        assert_eq!(telemetry.relu_nodes, 1);
        assert_eq!(telemetry.preactivations, 1);
        assert_eq!(telemetry.candidates, 1);
        assert_eq!(telemetry.collector_errors, 0);
    }

    #[test]
    fn phase_event_collector_enforces_attempt_cap_and_expired_deadline() {
        let model = BetaCrownModel::Graph(Box::new(build_phase_diag_graph()));
        let mut groups = HashMap::new();
        for (clause, upper) in [1.0f32, 1.5, 2.0].into_iter().enumerate() {
            let root = boxed(vec![0.0], vec![upper]);
            groups.insert(box_key(&root).unwrap(), (root, vec![clause]));
        }
        let mut clause_box = std::collections::BTreeMap::new();
        clause_box.insert(0usize, (0.0, 1.0));

        let capped = collect_phase_event_screen_telemetry(
            &model,
            &groups,
            std::slice::from_ref(&clause_box),
            1,
            Instant::now() + Duration::from_secs(1),
            1_000,
            Some(true),
        );
        assert_eq!(capped.probed_groups, 1, "cap counts all attempts");
        assert_eq!(capped.capped_groups, 2, "remaining roots are cap-skipped");
        assert_eq!(capped.collector_errors, 0);
        assert!(!capped.deadline_hit);

        let expired = collect_phase_event_screen_telemetry(
            &model,
            &groups,
            &[clause_box],
            8,
            Instant::now(),
            0,
            Some(true),
        );
        assert_eq!(expired.probed_groups, 0);
        assert!(expired.deadline_hit);
        assert_eq!(expired.deadline_capped_groups, 3);
        assert!(!expired.property_census_complete);
    }

    #[test]
    fn phase_event_gate_changes_no_verdict_or_partition() {
        let _env = ENV_LOCK.lock().unwrap();
        let attempts_before = PHASE_EVENT_TEST_ATTEMPTS.with(std::cell::Cell::get);
        let input = boxed(vec![0.0], vec![1.0]);
        let clauses = vec![vec![OutputConstraint::LessEqConst(0, -0.1)]];
        let mut clause_box = std::collections::BTreeMap::new();
        clause_box.insert(0usize, (0.0, 1.0));
        let boxes = vec![clause_box];

        let off_model = BetaCrownModel::Graph(Box::new(build_phase_diag_graph()));
        let off = ny_test_utils::env::with_serialized_env_vars(
            &[("NY_NN4SYS_1D_PHASE_EVENTS", "0")],
            || refine_clause_boxes_counted(&off_model, &input, &clauses, &boxes, None, None),
        );
        let on_model = BetaCrownModel::Graph(Box::new(build_phase_diag_graph()));
        let on = ny_test_utils::env::with_serialized_env_vars(
            &[
                ("NY_NN4SYS_1D_PHASE_EVENTS", "1"),
                ("NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS", "8"),
            ],
            || {
                refine_clause_boxes_counted_with_phase_target_override(
                    &on_model,
                    &input,
                    &clauses,
                    &boxes,
                    None,
                    None,
                    Some(true),
                )
            },
        );
        assert_eq!(
            PHASE_EVENT_TEST_ATTEMPTS.with(std::cell::Cell::get),
            attempts_before + 1,
            "eligible ON path must actually execute one point-JVP attempt"
        );
        assert_eq!(
            off, on,
            "diagnostic candidates must not reach verdict state"
        );
        assert_eq!(on, (vec![true], 1));
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
    fn build_plateau_graph() -> GraphNetwork {
        use ny_propagate::layers::{
            AddLayer, MulConstantLayer, ReLULayer, SigmoidLayer, SliceLayer, SubLayer,
        };
        let mut g = GraphNetwork::new();
        g.add_node(ny_propagate::GraphNode::from_input(
            "x0",
            Layer::Slice(SliceLayer::new(0, 0, 1)),
        ));
        g.add_node(ny_propagate::GraphNode::from_input(
            "x1",
            Layer::Slice(SliceLayer::new(0, 1, 2)),
        ));
        g.add_node(ny_propagate::GraphNode::binary(
            "s",
            Layer::Add(AddLayer),
            "x0",
            "x1",
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "r",
            Layer::ReLU(ReLULayer),
            vec!["s".to_string()],
        ));
        g.add_node(ny_propagate::GraphNode::binary(
            "plateau",
            Layer::Sub(SubLayer),
            "r",
            "s",
        ));
        g.add_node(ny_propagate::GraphNode::binary(
            "diff",
            Layer::Sub(SubLayer),
            "x0",
            "x1",
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "sig",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["diff".to_string()],
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "scaled",
            Layer::MulConstant(MulConstantLayer::new(ArrayD::from_elem(
                ndarray::IxDyn(&[1]),
                0.25f32,
            ))),
            vec!["sig".to_string()],
        ));
        g.add_node(ny_propagate::GraphNode::binary(
            "out",
            Layer::Add(AddLayer),
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
    fn build_mono_quadratic_graph() -> GraphNetwork {
        use ny_propagate::NETWORK_INPUT;
        let mut g = GraphNetwork::new();
        g.add_node(ny_propagate::GraphNode::new(
            "prod",
            Layer::MulBinary(ny_propagate::layers::MulBinaryLayer),
            vec![NETWORK_INPUT.to_string(), NETWORK_INPUT.to_string()],
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "out",
            Layer::Sub(ny_propagate::layers::SubLayer),
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

    #[test]
    fn finite_deadline_mvf_pilot_crown_gate_is_preshed_deterministically() {
        let mut enabled_gate =
            CrownCostGate::with_threshold_sample_probe(Some(CROWN_GATE_DEFAULT_MS), 1, true);
        enabled_gate.force_shed();
        assert!(!enabled_gate.should_attempt());
        let (attempts, _, shed, skips) = enabled_gate.stats();
        assert_eq!(attempts, 0);
        assert!(shed);
        assert_eq!(skips, 1);

        let mut explicitly_disabled_gate =
            CrownCostGate::with_threshold_sample_probe(None, 1, true);
        explicitly_disabled_gate.force_shed();
        assert!(
            explicitly_disabled_gate.should_attempt(),
            "NY_SCREEN_CROWN_MS=0 must remain authoritative"
        );
        let (attempts, _, shed, skips) = explicitly_disabled_gate.stats();
        assert_eq!(attempts, 0);
        assert!(!shed);
        assert_eq!(skips, 0);
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
                "NY_SCREEN_MVF_WAVE_SIZE",
                "NY_SCREEN_CELL_CHUNK",
                "NY_SCREEN_MVF_CHUNK",
            ] {
                env.remove(key);
            }
            assert_eq!(wave_size(), WAVE_SIZE);
            assert_eq!(mvf_deadline_wave_size(), F64_BATCH_MVF_DEADLINE_WAVE);
            assert_eq!(cell_chunk(), F64_BATCH_CELL_CHUNK);
            assert_eq!(mvf_chunk(), F64_BATCH_MVF_CHUNK);

            env.set("NY_SCREEN_WAVE_SIZE", " 17 ");
            env.set("NY_SCREEN_MVF_WAVE_SIZE", " 5 ");
            env.set("NY_SCREEN_CELL_CHUNK", "0");
            env.set("NY_SCREEN_MVF_CHUNK", "not-a-number");
            assert_eq!(wave_size(), 17);
            assert_eq!(mvf_deadline_wave_size(), 5);
            assert_eq!(cell_chunk(), F64_BATCH_CELL_CHUNK);
            assert_eq!(mvf_chunk(), F64_BATCH_MVF_CHUNK);
        });
    }

    /// Regression for the nn4sys dual root-wave starvation: under a finite
    /// deadline the expensive batch+MVF lane must expose a small, adjudicable
    /// wave instead of draining all 240 roots.  Other lanes and deadline-free
    /// runs retain the historical wide wave.
    #[test]
    fn finite_deadline_batched_mvf_uses_adjudicable_wave() {
        let _env = ENV_LOCK.lock().unwrap();
        ny_test_utils::env::with_env_edits(|env| {
            env.remove("NY_SCREEN_WAVE_SIZE");
            env.remove("NY_SCREEN_MVF_WAVE_SIZE");
            env.remove("NY_F64_BATCH_BOXES");
            env.remove("NY_F64_MVF");

            let dim = 1024usize;
            let w = ndarray::Array2::<f32>::from_elem((dim, dim), 0.001);
            let mut graph = GraphNetwork::new();
            graph.add_node(ny_propagate::GraphNode::from_input(
                "lin",
                Layer::Linear(ny_propagate::layers::LinearLayer::new(w, None).unwrap()),
            ));
            graph.set_output("lin");
            let model = BetaCrownModel::Graph(Box::new(graph));
            let mut esc = f64_leaf_escalation(&model).expect("f64 escalation");
            assert!(esc.batch && esc.mvf, "test must exercise batch+MVF");

            assert_eq!(
                scheduled_wave_len(
                    240,
                    Some(&esc),
                    Some(Instant::now() + Duration::from_secs(1)),
                    true,
                ),
                F64_BATCH_MVF_DEADLINE_WAVE,
                "finite deadline must not drain the huge root wave"
            );
            assert_eq!(
                scheduled_wave_len(240, Some(&esc), None, true),
                WAVE_SIZE.min(240),
                "deadline-free batch runs retain the throughput wave"
            );
            assert_eq!(
                scheduled_wave_len(
                    240,
                    Some(&esc),
                    Some(Instant::now() + Duration::from_secs(1)),
                    false,
                ),
                WAVE_SIZE.min(240),
                "non-dual batch+MVF screens retain the throughput wave"
            );

            esc.mvf = false;
            assert_eq!(
                scheduled_wave_len(
                    240,
                    Some(&esc),
                    Some(Instant::now() + Duration::from_secs(1)),
                    true,
                ),
                WAVE_SIZE.min(240),
                "cell-only batches retain the historical wave"
            );
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
        let mut fat = GraphNetwork::new();
        fat.add_node(ny_propagate::GraphNode::from_input(
            "lin",
            Layer::Linear(ny_propagate::layers::LinearLayer::new(w, None).unwrap()),
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
        let mut graph = GraphNetwork::new();
        graph.add_node(ny_propagate::GraphNode::from_input(
            "t",
            Layer::Tanh(ny_propagate::layers::TanhLayer),
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
        let mut g = GraphNetwork::new();
        g.add_node(ny_propagate::GraphNode::from_input(
            "l1",
            Layer::Linear(ny_propagate::layers::LinearLayer::new(w1, None).unwrap()),
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "relu",
            Layer::ReLU(ny_propagate::layers::ReLULayer),
            vec!["l1".to_string()],
        ));
        g.add_node(ny_propagate::GraphNode::new(
            "out",
            Layer::Linear(ny_propagate::layers::LinearLayer::new(w2, None).unwrap()),
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
