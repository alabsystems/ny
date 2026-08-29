// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #closure-adaptive: the lane picks its OWN closure mechanisms at runtime from
//! an observable it computes about itself.
//!
//! # The discriminant
//!
//! Measured 2026-08-22 on the 16 `cifar100_2024` rows this lane fails and
//! alpha-beta-CROWN proves: the ONLY variable that separates abc's wins from its
//! losses, with ZERO overlap, is the **per-batch child-closure fraction** —
//! wins 0.50-1.00, losses 0.18-0.34. Node count does not predict it (abc loses
//! at 33 618 nodes and wins at 16) and neither does the root bound (it wins at
//! post-alpha -0.765 and loses at -0.423). ny closes ZERO domains on 12 of the
//! 16: it is not inside abc's loss band, it is outside the distribution.
//!
//! That number is FREE to compute in a best-first BaB loop. `BabStats` already
//! carries `domains_created` and `closed`, and every open entry is by
//! construction not closed, so
//!
//! ```text
//!     closure = (closed_now - closed_at_mark) / (created_now - created_at_mark)
//! ```
//!
//! is the same quantity abc reports per batch, over a trailing window of this
//! lane's expansions. So the lane does not need to be CONFIGURED with closure
//! mechanisms by a human who guessed the benchmark; it can detect that its own
//! children are not closing and escalate.
//!
//! # The ladder, and why these rungs and not the obvious ones
//!
//! | tier | mechanism | why here |
//! |------|-----------|----------|
//! | 0 | base | nothing changed; a healthy row runs bit-identically |
//! | 1 | β split-constraint dual (`margin_row::beta`) | the ONLY mechanism measured to move this lane's closure at all: 0->2, 1->5, 8->22 closures (3-8x) on the failing rows, `accepted=true max_beta=1.643e-1` |
//! | 2 | β DEEPER — 8 dual-ascent trials per node instead of 1 | same mechanism class, more of it; the largest single closure movement in the whole matrix (idx_1298 0.116 -> 0.217) |
//! | 3 | Tier-2 EPOCH rebuild (`bab::try_epoch`) | last, because one attempt costs about a whole root build; the only mechanism that recomputes intermediates from scratch under the splits, which is abc's actual shape |
//!
//! ## Why Clip-and-Verify is NOT on this ladder
//!
//! It was, on the a-priori cost argument, and the measurement removed it. A
//! 24-run matrix on one pinned binary with engagement telemetry on every arm:
//!
//! ```text
//!   row        base    beta   beta+it8  clip    beta+clip
//!   idx_1298   0.014   0.116   0.217    0.014      -
//!   idx_4597   0.026   0.105   0.140    0.026    0.096
//!   idx_4169   0.000   0.000   0.013    0.000    0.000
//! ```
//!
//! `clip` is BIT-IDENTICAL to baseline — same expansion count, same closure
//! count, at all 20 sample points — and `beta+clip` is bit-identical to `beta`
//! alone. This is a real null with the mechanism PROVABLY ARMED, not the
//! vacuous kind: `[clip-interm] mean_rel_shrink` reads 8.4e-16 and 1.6e-4,
//! three-plus orders of magnitude too small to close a child.
//!
//! And there is a first-principles reason it cannot be rescued by dosing. A
//! split contributes ONE halfspace in 3072-dimensional input space. The
//! telemetry shows it cutting the SPLIT NEURON hard (layer 2 pos 19, upper
//! 0.1012 -> 0.0372, a 63% cut) and essentially nothing else (layer 0: 8.4e-16
//! across 13 of 13 neurons) — exactly what one halfspace does to neurons whose
//! weight vectors are near-orthogonal to its normal, which in high dimension is
//! almost all of them. So the flat dose-response measured twice was never about
//! the dose.
//!
//! That is also the reason the surviving rungs are what they are: **the
//! mechanism class that works here is dual/objective coupling, not box
//! tightening.** β couples the split constraint into the OBJECTIVE through the
//! backward pass rather than through intervals, and β is the only arm that moved
//! closure. A ladder whose upper rungs are box tighteners has no headroom no
//! matter how it is tuned. The clip code paths remain, on their own env gates,
//! and one sub-mechanism (`clip_tighten_pack`) is still unmeasured because the
//! y-pack LRU serves cached packs and it fires only ~4 times per run — but
//! neither is a rung until something changes the geometry.
//!
//! ## What the observable actually looks like on this lane
//!
//! Not a low rate — a BURST then a FLATLINE. idx_4597 closes 8 domains by
//! expansion 51 and then exactly 0 more across the next 260; idx_4169 has
//! `open == exp + 1` at all 19 sample points across 291 expansions. TERMINAL
//! closure is 0.000 in every arm including β. So ny is not sitting at the bottom
//! of abc's 0.18-0.34 loss band; it is in a different regime.
//!
//! This is why the observable is a TRAILING WINDOW and not a cumulative ratio.
//! A cumulative ratio keeps reporting the burst forever and would read a
//! flatlined search as healthy; a trailing window sees the flatline the moment
//! it starts. It is also why the threshold is not the load-bearing parameter on
//! these rows — at a terminal closure of 0.000 any threshold in (0, 0.5) fires.
//! What the threshold has to get right is the OTHER direction: NOT firing on the
//! rows this lane already proves. Measured, 0.40 does that with margin
//! (idx_5308 0.957, idx_7554 0.557-0.775, both held at base).
//!
//! # Soundness
//!
//! Every tier is sound on its own, INDEPENDENTLY of the policy, and the policy
//! never computes or publishes a bound:
//!
//! * **β, and β deep** — weak duality. For any `β >= 0` and any constraint
//!   `g_i <= 0` valid on the domain, `min_D f >= min_box (f + Σ β_i g_i)`.
//!   `β = 0` reproduces the base bound, so the multiplier set is never empty and
//!   the tier can only change WHICH valid dual point is scored. More ascent
//!   trials only search that same set harder — every trial is scored by the same
//!   certified pass and only an accepted improvement is kept. The extra
//!   arithmetic is charged into the same certified error lane
//!   (`apply_beta_terms`), and the head-seed shift refuses itself in outward
//!   mode when the seed carries no error lane.
//! * **Epoch rebuild** — the epoch gates are valid exactly on the split-halfspace
//!   intersection, which IS the popped domain, and the nested lane runs the same
//!   certified Outward machinery. The policy additionally requires
//!   `inherit_parent_intervals`, which is shrink-only against a parent frame
//!   valid over a superset region, and which REFUSES (fail-open to normal
//!   expansion) rather than closing anything if an interval crosses.
//! * **Escalating MID-SEARCH adds nothing.** A bound already pushed onto the
//!   heap was produced by the certified pass under the tier in force at the
//!   time, and every tier's bound is valid for the domain it was computed on.
//!   Raising the tier changes only what LATER evaluations receive as INPUT. No
//!   bound is ever recomputed-and-replaced, no cached bound is reinterpreted,
//!   and the child push rule (`cb = raw.max(b)`) is unchanged.
//! * **The certified pass stays the only scorer.** The policy reads two
//!   counters and writes one integer. It has no path to a published bound. The
//!   teeth cases are in `bab::closure_adaptive_tests`.
//!
//! # Hygiene (not soundness): the pack cache
//!
//! `MarginRowBab::lru` caches y-packs keyed by trunk-set. A pack built at one
//! tier and reused at another is merely LOOSER — sound, but it would make an
//! escalation read as inert on exactly the domains that triggered it. So a tier
//! change flushes the split-carrying packs. The empty-trunk root pack is
//! tier-invariant and is re-seeded rather than rebuilt.
//!
//! # Delivery
//!
//! One typed preset key, `margin_row.closure_adaptive`, default OFF — the POINT
//! is that the policy is automatic, so it is one key rather than six knobs a
//! human tunes per benchmark. `NY_MARGIN_ROW_CLOSURE_ADAPTIVE` is the
//! three-state research override (`1`/`0`/absent ⇒ defer to the preset), the
//! same contract `NY_MARGIN_ROW_K_ADAPT` has, because `run_instance.sh` exports
//! no `NY_*` and an env setting cannot fire in competition.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Base behaviour: nothing armed by the policy.
pub(crate) const TIER_BASE: usize = 0;
/// β split-constraint dual.
pub(crate) const TIER_BETA: usize = 1;
/// β again, but DEEPER: the dual ascent gets the full 8 trials per node instead
/// of 1. Same mechanism CLASS as tier 1, more of it.
pub(crate) const TIER_BETA_DEEP: usize = 2;
/// Dual ascent per node at [`TIER_BETA_DEEP`]. 8 is the reader's own clamp
/// ceiling and the value the measured `beta+iters8` arm used.
pub(crate) const BETA_DEEP_ITERS: usize = 8;
/// + Tier-2 EPOCH rebuild: fresh intermediates under the split constraints.
pub(crate) const TIER_EPOCH: usize = 3;
/// Top of the ladder.
pub(crate) const TIER_TOP: usize = TIER_EPOCH;

/// Trunk depth at which an escalated policy starts attempting epoch rebuilds.
///
/// 4 matches the depth the epoch mechanism was measured engaging at. Shallower
/// makes every early domain pay a root-build-sized rebuild; deeper and the tree
/// has usually already exploded.
pub(crate) const EPOCH_DEPTH_ON_ESCALATION: usize = 4;

/// Human name for a tier, used by every telemetry line so a log grep for
/// `[closure]` reads as prose rather than integers.
pub(crate) const fn tier_name(t: usize) -> &'static str {
    match t {
        TIER_BASE => "base",
        TIER_BETA => "beta",
        TIER_BETA_DEEP => "beta_deep",
        _ => "beta_deep+epoch",
    }
}

/// The live tier. Read on the hot path by `beta::enabled`,
/// `engine::clip_interm_enabled` and `bab::clip_tighten_enabled`; written only
/// from a BaB driver's SERIAL section (never from a rayon worker), so a
/// parallel frontier batch always sees one tier for its whole batch.
static TIER: AtomicUsize = AtomicUsize::new(TIER_BASE);

/// Typed preset route (`margin_row.closure_adaptive`), set once by the CLI
/// before any lane runs. Same bridge shape as `set_k_adaptive_preset`.
static PRESET: ny_levers::gate::BoolCell = ny_levers::gate::BoolCell::new();

/// Arm the typed preset route for the closure-adaptive policy.
pub fn set_closure_adaptive_preset(on: bool) {
    PRESET.set(on);
}

fn resolve_armed(preset: Option<bool>) -> bool {
    resolve_armed_from_layered(ny_levers::read_over_config(
        &ny_levers::decls::sound_channel::MARGIN_ROW_CLOSURE_ADAPTIVE,
        preset.map(ny_levers::LeverValue::Bool),
    ))
}

fn resolve_armed_from_layered(
    layered: Result<ny_levers::Resolved, ny_levers::ResolveError>,
) -> bool {
    layered
        .map(|resolved| resolved.value.as_bool())
        .unwrap_or(false)
}

/// Is the policy armed? Env override wins in BOTH directions; absence defers to
/// the preset (three-state, exactly like `NY_MARGIN_ROW_K_ADAPT`).
pub(crate) fn armed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = resolve_armed(PRESET.get());
        if on {
            // R9. An armed run announces itself ONCE even if no tree ever runs,
            // so "the policy never printed" is always distinguishable from "the
            // policy printed and did nothing".
            eprintln!(
                "[closure] policy ARMED: warmup={} window={} thresh={:.2} max_tier={} \
                 ({})",
                warmup(),
                window(),
                threshold(),
                max_tier(),
                tier_name(max_tier()),
            );
        }
        on
    })
}

/// The live tier (`TIER_BASE` when the policy is off or has not escalated).
#[inline]
pub(crate) fn tier() -> usize {
    TIER.load(Ordering::Relaxed)
}

/// Dual-ascent trials per node the LADDER is asking for, or `0` when it is
/// asking for nothing. `beta::iters` takes the max of this and its own
/// env/default value, so the ladder can only ever ask for MORE ascent — it can
/// never quietly weaken a configured research arm.
#[inline]
pub(crate) fn beta_iters_floor() -> usize {
    if tier() >= TIER_BETA_DEEP {
        BETA_DEEP_ITERS
    } else {
        0
    }
}

/// Reset the ladder. Called at every lane entry so one instance's escalation
/// cannot leak into the next tree in the same process (the classwise schedule
/// runs several trees, and a sweep runs several rows).
pub(crate) fn reset() {
    TIER.store(TIER_BASE, Ordering::Relaxed);
}

/// Closure fraction at or below which the policy escalates.
///
/// DEFAULT 0.40, and the justification is abc's own data rather than a taste
/// judgement: on the 16-row calibration set abc's LOSSES top out at 0.34 and its
/// WINS start at 0.50, an empty band. 0.40 sits strictly inside it, so the rule
/// reproduces abc's win/loss partition with zero misclassification in either
/// direction — strictly above every loss means every row abc fails would
/// escalate, strictly below every win means no row abc proves would.
pub(crate) fn threshold() -> f64 {
    static T: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        let threshold = crate::preset_gates::margin_row_closure_thresh();
        if threshold.is_finite() && threshold > 0.0 && threshold < 1.0 {
            threshold
        } else {
            0.40
        }
    })
}

/// Expansions before the FIRST decision.
///
/// DEFAULT 24. abc proves the three calibration rows in 16 / 48 / 48 nodes, so a
/// warm-up shorter than ~16 cannot tell "no closure yet" from "has not finished
/// its first batch". 24 = 1.5x the smallest abc proof: by then a row on a
/// winning trajectory has discharged a full batch and started another. At this
/// lane's measured ~3 expansions/s that is ~8 s of a 100 s budget, which bounds
/// what the observation itself can cost.
pub(crate) fn warmup() -> usize {
    static W: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *W.get_or_init(|| crate::preset_gates::margin_row_closure_warmup().clamp(1, 100_000))
}

/// Expansions per observation window after the warm-up.
///
/// DEFAULT 16 — about 32 children, i.e. abc's batch scale, and ~5 s at this
/// lane's rate. Frequent enough that the ladder can be climbed to the top
/// inside a 100 s budget (3 escalations = 48 expansions ≈ 16 s), coarse enough
/// that one lucky or unlucky expansion cannot flip the tier.
pub(crate) fn window() -> usize {
    static W: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *W.get_or_init(|| crate::preset_gates::margin_row_closure_window().clamp(1, 100_000))
}

/// Highest tier the policy may reach. Research knob for single-mechanism arms
/// (`=1` isolates β, `=2` adds clip-interm); default is the whole ladder.
pub(crate) fn max_tier() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| crate::preset_gates::margin_row_closure_max_tier().min(TIER_TOP))
}

/// What a window's observation asks the driver to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// Not a decision point (policy off, or mid-window).
    Idle,
    /// Observed; closure is healthy or the ladder is exhausted. Nothing changed.
    Hold,
    /// Tier changed to this value; the driver must flush split-carrying packs.
    Escalate(usize),
    /// The last escalation neither bought closure nor kept up: reverted to this
    /// tier and the ladder is frozen.
    Rollback(usize),
}

/// Per-tree closure observer.
///
/// Owned by the BaB driver, so a nested epoch subtree gets its own instance
/// while sharing the process-global tier (an escalation earned by the parent
/// stays in force inside the epoch, which is the intent).
#[derive(Debug)]
pub(crate) struct ClosureMonitor {
    on: bool,
    warmup: usize,
    window: usize,
    thresh: f64,
    max_tier: usize,
    /// Expansion count at the last observation.
    mark_exp: usize,
    mark_created: usize,
    mark_closed: usize,
    mark_secs: f64,
    /// Closure and per-expansion cost measured in the window that triggered the
    /// most recent escalation, kept so the rollback test has something to
    /// compare against.
    pre_esc_closure: f64,
    pre_esc_secs_per_exp: f64,
    /// Set once a rollback has fired: the ladder is frozen for the rest of the
    /// tree.
    frozen: bool,
    /// Observations emitted (telemetry sequence number).
    obs: usize,
    /// Escalations performed by THIS tree.
    escalations: usize,
}

/// Fraction of the lane's wall clock after which the policy stops escalating.
///
/// The kill criterion is "never spend so much that it loses a row it would
/// otherwise prove". A fresh escalation needs at least two more windows before
/// its effect is even observable, so buying one in the last quarter of the
/// budget can only pay a cost it cannot collect on.
const NO_ESCALATE_AFTER_FRAC: f64 = 0.75;

/// Windows of remaining budget a fresh escalation must be able to afford, at
/// the CURRENT measured per-expansion cost and assuming the escalation makes
/// each expansion this much dearer.
const ESCALATION_COST_MULTIPLE: f64 = 3.0;

/// Rollback trigger: the escalation is reverted when the window after it both
/// failed to raise closure and ran this many times slower per expansion.
const ROLLBACK_SLOWDOWN: f64 = 4.0;

impl ClosureMonitor {
    /// Build the observer for one tree. `on` is the policy arming; the caller
    /// passes it explicitly so tests can drive the monitor without touching
    /// process-global state.
    pub(crate) fn new(on: bool) -> Self {
        Self {
            on,
            warmup: warmup(),
            window: window(),
            thresh: threshold(),
            max_tier: max_tier(),
            mark_exp: 0,
            mark_created: 0,
            mark_closed: 0,
            mark_secs: 0.0,
            pre_esc_closure: f64::NAN,
            pre_esc_secs_per_exp: f64::NAN,
            frozen: false,
            obs: 0,
            escalations: 0,
        }
    }

    /// Test seam: construct with explicit policy parameters.
    #[cfg(test)]
    pub(crate) fn with_params(warmup: usize, window: usize, thresh: f64, max_tier: usize) -> Self {
        Self {
            on: true,
            warmup,
            window,
            thresh,
            max_tier,
            ..Self::new(false)
        }
    }

    #[cfg(test)]
    pub(crate) const fn escalations(&self) -> usize {
        self.escalations
    }

    /// Observe one step of the tree.
    ///
    /// Called from the driver's SERIAL section with the live counters. Returns
    /// what the driver must do; the tier itself is already stored when
    /// `Escalate`/`Rollback` is returned.
    ///
    /// * `expansions`/`created`/`closed` — `BabStats` counters, live.
    /// * `open` — open domains (heap length, plus anything in flight).
    /// * `elapsed` / `remaining` — seconds spent in the tree so far and
    ///   seconds left before the lane deadline (`None` = no deadline).
    pub(crate) fn observe(
        &mut self,
        expansions: usize,
        created: usize,
        closed: usize,
        open: usize,
        elapsed: f64,
        remaining: Option<f64>,
    ) -> Decision {
        if !self.on {
            return Decision::Idle;
        }
        let due = if self.obs == 0 {
            expansions >= self.warmup
        } else {
            expansions >= self.mark_exp + self.window
        };
        if !due {
            return Decision::Idle;
        }
        let d_created = created.saturating_sub(self.mark_created);
        let d_closed = closed.saturating_sub(self.mark_closed);
        let d_exp = expansions.saturating_sub(self.mark_exp).max(1);
        let d_secs = (elapsed - self.mark_secs).max(0.0);
        // The observable. Of the domains created in this window, what fraction
        // got closed. Identical in meaning to abc's per-batch child-closure
        // fraction, which is the variable that separates its wins from its
        // losses with zero overlap.
        let win_closure = if d_created == 0 {
            // No children created in the window (root closed, or every pop was
            // discharged before expanding). Not evidence of a stuck search.
            1.0
        } else {
            d_closed as f64 / d_created as f64
        };
        let cum_closure = if created == 0 {
            1.0
        } else {
            closed as f64 / created as f64
        };
        let secs_per_exp = d_secs / d_exp as f64;
        self.obs += 1;
        let cur = tier();

        // --- rollback test, before anything else can escalate again ---
        if !self.frozen
            && self.escalations > 0
            && cur > TIER_BASE
            && self.pre_esc_closure.is_finite()
            && self.pre_esc_secs_per_exp.is_finite()
            && self.pre_esc_secs_per_exp > 0.0
            && win_closure <= self.pre_esc_closure
            && secs_per_exp > ROLLBACK_SLOWDOWN * self.pre_esc_secs_per_exp
        {
            let back = cur - 1;
            TIER.store(back, Ordering::Relaxed);
            self.frozen = true;
            eprintln!(
                "[closure] obs={} exp={expansions} open={open} closure={win_closure:.3} \
                 (cum {cum_closure:.3}) ROLLBACK tier {cur}->{back} ({} -> {}): \
                 closure did not rise ({:.3} -> {win_closure:.3}) and cost/expansion went \
                 {:.3}s -> {secs_per_exp:.3}s (>{ROLLBACK_SLOWDOWN:.0}x); ladder frozen",
                self.obs,
                tier_name(cur),
                tier_name(back),
                self.pre_esc_closure,
                self.pre_esc_secs_per_exp,
            );
            self.mark(expansions, created, closed, elapsed);
            return Decision::Rollback(back);
        }

        // --- the decision ---
        let healthy = win_closure >= self.thresh;
        let at_top = cur >= self.max_tier;
        let late = remaining.is_some_and(|r| {
            let total = elapsed + r;
            total > 0.0 && elapsed / total > NO_ESCALATE_AFTER_FRAC
        });
        // A fresh escalation must be able to afford at least two more windows
        // at ESCALATION_COST_MULTIPLE the current per-expansion cost, or it can
        // only pay a bill it will never collect on.
        let need_secs = ESCALATION_COST_MULTIPLE * 2.0 * self.window as f64 * secs_per_exp;
        let unaffordable = remaining.is_some_and(|r| r < need_secs);

        if healthy || at_top || self.frozen || late || unaffordable {
            let why = if healthy {
                "healthy".to_string()
            } else if self.frozen {
                "ladder frozen after rollback".to_string()
            } else if at_top {
                format!("already at max tier {} ({})", cur, tier_name(cur))
            } else if late {
                format!(
                    "past {:.0}% of the lane ({elapsed:.1}s spent, {:.1}s left): a fresh tier \
                     cannot pay for itself",
                    NO_ESCALATE_AFTER_FRAC * 100.0,
                    remaining.unwrap_or(f64::NAN),
                )
            } else {
                format!(
                    "unaffordable: needs {need_secs:.1}s at {secs_per_exp:.3}s/exp, \
                     {:.1}s left",
                    remaining.unwrap_or(f64::NAN),
                )
            };
            eprintln!(
                "[closure] obs={} exp={expansions} open={open} created+{d_created} closed+{d_closed} \
                 closure={win_closure:.3} (cum {cum_closure:.3}, thresh {:.2}) \
                 {:.3}s/exp tier={cur} ({}) HOLD: {why}",
                self.obs,
                self.thresh,
                secs_per_exp,
                tier_name(cur),
            );
            self.mark(expansions, created, closed, elapsed);
            return Decision::Hold;
        }

        let next = cur + 1;
        TIER.store(next, Ordering::Relaxed);
        self.escalations += 1;
        self.pre_esc_closure = win_closure;
        self.pre_esc_secs_per_exp = secs_per_exp;
        eprintln!(
            "[closure] obs={} exp={expansions} open={open} created+{d_created} closed+{d_closed} \
             closure={win_closure:.3} (cum {cum_closure:.3}) < thresh {:.2} \
             {secs_per_exp:.3}s/exp {:.1}s left ESCALATE tier {cur}->{next} ({} -> {}): \
             children are not closing",
            self.obs,
            self.thresh,
            remaining.unwrap_or(f64::INFINITY),
            tier_name(cur),
            tier_name(next),
        );
        self.mark(expansions, created, closed, elapsed);
        Decision::Escalate(next)
    }

    /// Final receipt for the tree. Always printed when the policy is armed, so
    /// a run that observed nothing is still distinguishable from a run where
    /// the policy never loaded (R9).
    pub(crate) fn finish(&self, expansions: usize, created: usize, closed: usize, stop: &str) {
        if !self.on {
            return;
        }
        let cum = if created == 0 {
            f64::NAN
        } else {
            closed as f64 / created as f64
        };
        eprintln!(
            "[closure] TREE DONE stop={stop} exp={expansions} created={created} closed={closed} \
             closure={cum:.3} observations={} escalations={} final_tier={} ({})",
            self.obs,
            self.escalations,
            tier(),
            tier_name(tier()),
        );
    }

    fn mark(&mut self, expansions: usize, created: usize, closed: usize, elapsed: f64) {
        self.mark_exp = expansions;
        self.mark_created = created;
        self.mark_closed = closed;
        self.mark_secs = elapsed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that move the process-global tier.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn env_wins_both_directions_and_malformed_suppresses_the_closure_preset() {
        let resolve = |raw: Option<&str>, preset: Option<bool>| {
            let raw = raw.map(str::to_owned);
            resolve_armed_from_layered(ny_levers::read_over_config_with(
                &ny_levers::decls::sound_channel::MARGIN_ROW_CLOSURE_ADAPTIVE,
                move |_| raw,
                preset.map(ny_levers::LeverValue::Bool),
            ))
        };
        assert!(resolve(None, Some(true)));
        assert!(!resolve(None, Some(false)));
        assert!(!resolve(Some("0"), Some(true)));
        assert!(resolve(Some("1"), Some(false)));
        assert!(
            !resolve(Some("true"), Some(true)),
            "malformed env must land on the declaration's unset/off default"
        );
    }

    #[test]
    fn a_disarmed_monitor_is_inert_and_never_moves_the_tier() {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let mut m = ClosureMonitor::new(false);
        for e in 0..500 {
            assert_eq!(
                m.observe(e, 2 * e, 0, e, e as f64, Some(1000.0)),
                Decision::Idle
            );
        }
        assert_eq!(tier(), TIER_BASE);
        assert_eq!(m.escalations(), 0);
    }

    #[test]
    fn a_healthy_row_holds_at_base() {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let mut m = ClosureMonitor::with_params(24, 16, 0.40, TIER_TOP);
        // One child of every two closes: closure 0.5, abc's win floor.
        for e in 0..400 {
            let d = m.observe(e, 2 * e, e, e, 0.3 * e as f64, Some(1000.0));
            assert!(matches!(d, Decision::Idle | Decision::Hold), "{d:?}");
        }
        assert_eq!(tier(), TIER_BASE, "a healthy row must not escalate");
        assert_eq!(m.escalations(), 0);
    }

    #[test]
    fn a_zero_closure_row_climbs_the_whole_ladder_and_then_holds() {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let mut m = ClosureMonitor::with_params(24, 16, 0.40, TIER_TOP);
        let mut seen = Vec::new();
        for e in 0..400 {
            // Zero closure, constant cost: the measured ny signature.
            match m.observe(e, 2 * e, 0, e, 0.3 * e as f64, Some(1000.0)) {
                Decision::Escalate(t) => seen.push(t),
                Decision::Rollback(t) => panic!("unexpected rollback to {t}"),
                _ => {}
            }
        }
        assert_eq!(seen, vec![TIER_BETA, TIER_BETA_DEEP, TIER_EPOCH]);
        assert_eq!(tier(), TIER_TOP);
        assert_eq!(m.escalations(), 3);
    }

    #[test]
    fn abcs_loss_band_escalates_and_its_win_band_does_not() {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // The threshold's whole justification is that it separates abc's two
        // measured bands. Pin both edges.
        for (closed_per_100, band, want_escalate) in [
            (18usize, "abc loss floor", true),
            (34, "abc loss ceiling", true),
            (50, "abc win floor", false),
            (100, "abc win ceiling", false),
        ] {
            reset();
            let mut m = ClosureMonitor::with_params(24, 16, 0.40, TIER_TOP);
            let mut escalated = false;
            for e in 0..200 {
                // created = 100 per expansion, closed = closed_per_100 per
                // expansion: an exact rational closure fraction.
                if let Decision::Escalate(_) = m.observe(
                    e,
                    100 * e,
                    closed_per_100 * e,
                    e,
                    0.3 * e as f64,
                    Some(1000.0),
                ) {
                    escalated = true;
                }
            }
            assert_eq!(escalated, want_escalate, "{band}");
        }
        reset();
    }

    #[test]
    fn the_ladder_stops_when_the_budget_cannot_pay_for_it() {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let mut m = ClosureMonitor::with_params(24, 16, 0.40, TIER_TOP);
        // Zero closure, but only 1 s left and 1 s/expansion: the escalation
        // needs 3 * 2 * 16 * 1 = 96 s it does not have.
        for e in 0..200 {
            let d = m.observe(e, 2 * e, 0, e, e as f64, Some(1.0));
            assert!(matches!(d, Decision::Idle | Decision::Hold), "{d:?}");
        }
        assert_eq!(tier(), TIER_BASE);
        reset();
    }

    #[test]
    fn the_ladder_stops_in_the_last_quarter_of_the_lane() {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let mut m = ClosureMonitor::with_params(24, 16, 0.40, TIER_TOP);
        // 90 s spent, 10 s left: 90% of the lane, past NO_ESCALATE_AFTER_FRAC.
        // Cheap expansions, so affordability is not what refuses this.
        for e in 0..200 {
            let d = m.observe(e, 2 * e, 0, e, 90.0, Some(10.0));
            assert!(matches!(d, Decision::Idle | Decision::Hold), "{d:?}");
        }
        assert_eq!(tier(), TIER_BASE);
        reset();
    }

    #[test]
    fn an_escalation_that_only_costs_is_rolled_back_and_the_ladder_freezes() {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let mut m = ClosureMonitor::with_params(8, 8, 0.40, TIER_TOP);
        // Window 1: zero closure at 0.1 s/expansion -> escalate to tier 1.
        let d = m.observe(8, 16, 0, 16, 0.8, Some(1000.0));
        assert_eq!(d, Decision::Escalate(TIER_BETA));
        // Window 2: still zero closure, and 10x dearer -> roll back and freeze.
        let d = m.observe(16, 32, 0, 32, 0.8 + 8.0, Some(1000.0));
        assert_eq!(d, Decision::Rollback(TIER_BASE));
        assert_eq!(tier(), TIER_BASE);
        // Frozen: no further escalation however bad closure gets.
        for e in 24..200 {
            let d = m.observe(e, 2 * e, 0, e, 8.8 + 0.1 * e as f64, Some(1000.0));
            assert!(matches!(d, Decision::Idle | Decision::Hold), "{d:?}");
        }
        assert_eq!(tier(), TIER_BASE);
        reset();
    }

    #[test]
    fn max_tier_caps_the_ladder_for_single_mechanism_arms() {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let mut m = ClosureMonitor::with_params(8, 8, 0.40, TIER_BETA);
        for e in 0..200 {
            let _ = m.observe(e, 2 * e, 0, e, 0.1 * e as f64, Some(1000.0));
        }
        assert_eq!(tier(), TIER_BETA);
        reset();
    }

    #[test]
    fn tier_names_cover_the_ladder() {
        assert_eq!(tier_name(TIER_BASE), "base");
        assert_eq!(tier_name(TIER_BETA), "beta");
        assert_eq!(beta_iters_floor(), 0);
        assert_eq!(tier_name(TIER_BETA_DEEP), "beta_deep");
        assert_eq!(tier_name(TIER_EPOCH), "beta_deep+epoch");
    }
}
