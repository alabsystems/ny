// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #backward-interm (`NY_MARGIN_ROW_BACKWARD_INTERM=1`, default OFF):
//! backward-computed root intermediates for the margin-row lane (#twinwall).
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
//! it. v1 runs on the ROOT build only (`splits.is_empty()`), never inside
//! Tier-2 epoch rebuilds, so one instance pays the budget at most once.

use std::time::Instant;

use ndarray::Array2;

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
    /// Max unstable neurons re-derived per layer (widest first).
    pub topk: usize,
}

impl BiCfg {
    /// Test-friendly default mirror of the env defaults.
    pub(crate) fn defaults() -> Self {
        Self {
            secs: 20.0,
            chunk: 256,
            topk: 1024,
        }
    }
}

/// Env resolution. Exact `"1"` arms; read per BUILD (once per instance, no
/// `OnceLock`) so sealed A/Bs and unit tests can flip arms without process
/// restarts. Disarmed ⇒ the root build is byte-identical to its history.
pub(crate) fn from_env(mode: RoundMode) -> Option<BiCfg> {
    if !mode.outward() {
        // Parity is the bit-parity oracle against core.py; it must not move.
        return None;
    }
    // Env wins in BOTH directions where present (research override); absent,
    // the typed preset route decides (`margin_row.backward_interm` — env
    // cannot fire in competition, `run_instance.sh` exports exactly one NY_*).
    //
    // That three-way rule IS `read_over_config`'s layering, so it is expressed
    // through the chokepoint rather than restated here: exact "1"/"0" win over
    // the preset; a PRESENT near-miss token ("true", " 1", "") is a recorded
    // rejection that suppresses the preset and lands on the declaration's
    // `false` default — the old `Some(_) => return None` arm, now visible in
    // the receipt; absence falls through to the preset. Fails closed to dark.
    let armed = ny_levers::read_over_config(
        &ny_levers::decls::margin_row::MARGIN_ROW_BACKWARD_INTERM,
        Some(ny_levers::LeverValue::Bool(super::backward_interm_preset())),
    )
    .map(|resolved| resolved.value.as_bool())
    .unwrap_or(false);
    if !armed {
        return None;
    }
    let d = BiCfg::defaults();
    let secs = ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_BI_SECS)
        .value
        .as_f64()
        .unwrap_or(d.secs);
    // The clamps stay at the reader: the chokepoint hands an explicit `0` back
    // instead of swallowing it, so "explicitly zero" and "absent" remain
    // different facts for the receipt.
    let chunk = ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_BI_CHUNK)
        .value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(d.chunk)
        .clamp(1, 4096);
    let topk = ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_BI_TOPK)
        .value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(d.topk)
        .max(1);
    Some(BiCfg { secs, chunk, topk })
}

/// Per-build state: budget clock + telemetry accumulators.
pub(crate) struct BiState {
    cfg: BiCfg,
    t0: Instant,
    /// Effective wall budget (cfg.secs capped to 40% of the remaining
    /// deadline at construction; the tree search is never starved).
    budget: f64,
    deadline: Option<Instant>,
    exhausted: bool,
    /// Totals across layers (printed per layer; summed for the final line).
    tightened_total: usize,
    stabilized_total: usize,
    crossed_total: usize,
    pass_errors: usize,
}

impl BiState {
    pub(crate) fn new(cfg: BiCfg, deadline: Option<Instant>) -> Self {
        let t0 = Instant::now();
        let budget = match deadline {
            Some(dl) => cfg
                .secs
                .min(0.4 * dl.saturating_duration_since(t0).as_secs_f64()),
            None => cfg.secs,
        };
        // Engagement telemetry (R9): armed must be visible in one log line
        // even when every layer is later skipped.
        eprintln!(
            "[backward-interm] armed: budget={budget:.1}s chunk={} topk={}",
            cfg.chunk, cfg.topk
        );
        Self {
            cfg,
            t0,
            budget,
            deadline,
            exhausted: false,
            tightened_total: 0,
            stabilized_total: 0,
            crossed_total: 0,
            pass_errors: 0,
        }
    }

    fn out_of_time(&mut self) -> bool {
        if self.exhausted {
            return true;
        }
        if self.t0.elapsed().as_secs_f64() > self.budget {
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
        if self.out_of_time() {
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
        let unst_before = sel.len();
        sel.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        sel.truncate(self.cfg.topk);
        // Ascending neuron order for cache-friendly seed construction.
        let mut cols: Vec<usize> = sel.into_iter().map(|(_, j)| j).collect();
        cols.sort_unstable();

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
        };
        let (mut tightened, mut stabilized, mut crossed, mut done) =
            (0usize, 0usize, 0usize, 0usize);
        {
            let eng = BackwardEngine::new(net, &tmp);
            let n_pre = l.len();
            'chunks: for chunk in cols.chunks(self.cfg.chunk) {
                if self.out_of_time() {
                    break;
                }
                let w = chunk.len();
                let mut s = Array2::<f64>::zeros((n_pre, w));
                for (c, &j) in chunk.iter().enumerate() {
                    s[[j, c]] = 1.0;
                }
                // Identity seed is exact: the engine zero-fills its error lane.
                let seed = Seed { s, e: None };
                let (lb, ub) = match (
                    eng.run_prefix(&seed, relu_op, LaneDir::Lower),
                    eng.run_prefix(&seed, relu_op, LaneDir::Upper),
                ) {
                    (Ok(plo), Ok(pup)) => (eng.concretize_lower(&plo), eng.concretize_upper(&pup)),
                    _ => {
                        // Fail-open: an engine refusal here costs tightness
                        // only. Stop the layer (a failure would repeat).
                        self.pass_errors += 1;
                        break 'chunks;
                    }
                };
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
            }
        }
        *layers = tmp.layers;
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
    pub(crate) fn finish(&self) {
        eprintln!(
            "[backward-interm] done: tightened={} stabilized={} crossed={} pass_errors={} \
secs={:.2} exhausted={}",
            self.tightened_total,
            self.stabilized_total,
            self.crossed_total,
            self.pass_errors,
            self.t0.elapsed().as_secs_f64(),
            self.exhausted
        );
    }
}
