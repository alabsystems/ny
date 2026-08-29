// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #margin-row-domain-alpha (`NY_MARGIN_ROW_DOMAIN_ALPHA` /
//! `margin_row.domain_alpha`, default OFF): PER-DOMAIN alpha for the
//! margin-row lane (#twinwall).
//!
//! WHY. ny optimizes the lower-line slope alpha exactly ONCE, at the root
//! (`alpha_opt::maybe_optimize_root_alpha`, the only non-test call site), and
//! then freezes it (`mod.rs`: `let root = root; // freeze`). Every child domain
//! runs `engine::domain_gates_split_only`, which CLONES the root's
//! `alpha/s/c/ms` and overrides only the split neuron's index. At depth 30
//! thirty neurons carry exact fixed lines and every OTHER unstable neuron is
//! still relaxed with a slope optimized for the whole ROOT box. That deficit
//! grows with depth by construction, which is the shape the measured wall
//! needs: the gating (frontier-worst) bound at a given depth is BIT-IDENTICAL
//! across every arm tried so far (base, β, β×8, stateless clip, stateful clip,
//! both branching reductions), improving ~0.0096-0.0120 per level against a
//! gap of 0.155-0.203 — 13-21 more levels, i.e. 2^13..2^21 domains where the
//! budget affords ~90. Search cannot win; the per-level GAIN has to move.
//!
//! WHY IT IS PAIRED WITH CLIP, and why the clip null does not retire clip.
//! alpha-beta-CROWN ships clip and per-domain alpha as a COMPOUNDING pair:
//! clip tightens a domain's stored intermediate `(l, u)` -> those bounds define
//! the ReLU relaxation TRIANGLES -> alpha is RE-OPTIMIZED per domain against
//! the tightened triangles -> the better slope lets the next clip round cut
//! deeper. ny had the first half only, so a 1.6e-4 box shrink had NO CONSUMER
//! and stayed 1.6e-4 forever (measured: stateful clip carried 18-24 neurons to
//! generation 9 with `shrink_mean` 1.5-2.7e-2 and the gating curve came out
//! bit-identical, 20/20 points on idx_4597 and 8/8 on idx_1298). This module is
//! the consumer. It runs on the gates `domain_gates` already produced —
//! INCLUDING `clip_tighten_gates`' rebuilt `(alpha, s, c)` — so an armed clip
//! feeds it directly. MEASURE THE 2x2 (off / clip / alpha / BOTH); measuring
//! the pair one-at-a-time is guaranteed to null twice.
//!
//! THE GRADIENT ALREADY EXISTS, and it is not the banked sign-definite one.
//! `alpha_opt::supergradient` computes, for the frozen-sign linearization,
//!
//! ```text
//!   d bound_col / d alpha_{li,j} = vp_{li,j,col} * U_{li,j,col}
//! ```
//!
//! with `vp = max(v, 0) >= 0` the pass's own pre-transform incoming coefficient
//! and `U` the linearized-walk value of neuron `j`'s pre-activation at the
//! concretize argmin `x*`. `U` is the pre-activation of an UNSTABLE neuron, so
//! it straddles zero and the product is genuinely signed — this is a real
//! direction, unlike the banked internal-verifier finding (`α`-CROWN's
//! `alpha_grad`, sign-definite `<= 0` and clamped to 0, Lean
//! `alpha_sound_regardless`), which is a different lane and does not apply
//! here. `beta.rs` reuses the SAME walk (`step_scales`) for its split duals, so
//! the ascent-trial shape below is not a fallback for a missing gradient: it is
//! the established lane pattern — propose from the linearization, SCORE with
//! the unchanged certified pass, accept monotonically.
//!
//! WHY NOT THE FREE ARM. `9e695ba92` measured, in the MULTI-OBJECTIVE joint
//! lane (`beta_crown/state/alpha/graph_runtime.rs`, `NY_MO_ALPHA_HEURISTIC`),
//! that merely RE-DERIVING the current-box heuristic `alpha = [u > -l]` beat
//! the stored ancestor-optimized alpha by +0.34 at depth 16 — staleness costing
//! more than optimization gains, at zero passes. That is a real result and it
//! does NOT transfer here, for a checkable reason rather than an opinion:
//! `engine::domain_gates` is `domain_gates_split_only` + `clip_tighten_gates`
//! and nothing else writes a per-domain trunk alpha, so for every UNSPLIT
//! neuron this domain's `(l, u)` is bit-identical to the root's — and the
//! root's alpha came from `gates_from_box` on exactly those `(l, u)`. "Fresh"
//! and "stored" are the same number. Where the box DOES move, clip (and the
//! Tier-2 epoch refresh) already re-derive `gates_from_box` +
//! `repair_upper_lines`, i.e. the probe's intervention is already implemented
//! on the only paths that could need it. `margin_row::tests::
//! fresh_current_box_heuristic_is_a_noop_in_this_lane` pins that executably.
//! So this lane's stale alpha is not a REFRESH problem, it is an
//! OPTIMIZATION-never-happened problem, and only the optimizer arm can move it.
//!
//! WARM START (abc's `domain_updater.py:123-128`). A child inherits its
//! parent's ACCEPTED alpha as a sparse `(layer, neuron, alpha)` list and
//! applies it before its own first pass, so the tightening compounds down the
//! tree instead of resetting at every node. Inheritance is sound for the same
//! reason the initial value is (below): alpha validity is box-independent.
//!
//! SOUNDNESS. Tightness-only, and the argument is short:
//!
//! * `alpha*y <= relu(y)` holds POINTWISE ON ALL OF R for ANY `alpha in
//!   [0, 1]`. Validity of the lower line is box-independent, so it does not
//!   matter which domain, which box, or which ancestor produced the value.
//!   Every write here clamps to `[0, 1]` ([`write_alphas`]).
//! * The upper chord `(s, c)` depends only on `(l, u)`, which this module
//!   never touches — the build's (or clip's) `repair_upper_lines` certificate
//!   stands verbatim.
//! * THE ONE LOAD-BEARING COUPLING: the backward error lane contracts by
//!   `ms = max(alpha, s)`, the Lipschitz constant of `v -> vp*a + vn*s`. A
//!   stale `ms` under a RAISED alpha would UNDER-scale the carried error — a
//!   false-UNSAT generator. `ms` is re-derived from the written alpha on EVERY
//!   write, by the same `max(alpha, s)` formula the root uses.
//! * The forward tableau is NOT rebuilt: the stored `l`/`u`/`clip_rows` were
//!   produced by a valid relaxation and remain valid enclosures of the true
//!   pre-activation ranges regardless of which VALID backward alpha is used
//!   later. `apply_gates`' outward widening (whose historical constant assumes
//!   a binary lower product) is a FORWARD site and is never entered from here.
//! * EXACT LINES ARE NEVER TOUCHED. [`movable`] admits a neuron only when the
//!   domain's own `s` is strictly inside `(0, 1)` — which is exactly "this
//!   neuron carries a relaxation triangle here". Stable-active (`s = 1`),
//!   stable-inactive (`s = 0`), split-baked (`s in {0, 1}`) and
//!   clip-stabilized neurons all fail that test, and split-pinned positions are
//!   excluded explicitly on top of it.
//! * THE CERTIFIED PASS IS THE ONLY SCORER. A candidate is accepted only on a
//!   measured improvement of the same certified composite the domain verdict
//!   uses, monotone keep-best exactly as `beta.rs` does. A bad optimizer costs
//!   one pass and some tightness; it cannot manufacture a proof. On rejection
//!   the gates are restored to the last ACCEPTED vector, so the `dom` stored in
//!   `NodeState` always describes the bound that was published.
//!
//! COST. One certified pass per ascent trial per expanded domain — the same
//! cost model as `#margin-row-beta`, whose default is likewise 1 trial. The
//! MEASURED column-scaling fit (docs/MARGIN_ROW_COLUMN_SCALING_2026-08-23.md)
//! is `pass_ms ~= 60 + 2.4*R` on resnet_medium, so a lane pass at the
//! measured `nf` of 3-11 failing columns costs ~90-130 ms and the default 2
//! trials cost ~0.2-0.3 s/domain. The warm start is FREE (applying a
//! per-domain gate override measures a 0.98-1.12x pass-cost ratio at every
//! width), which is what makes low-iteration + inheritance the affordable
//! shape: the alpha keeps improving down the tree without paying for it again
//! at each node.
//!
//! MEASURED 2026-08-24 (rev 4921 = this tree exactly, `dirty=false`; pinned
//! binary sha256 10da1548..., box-locked, official budgets, 3 runs/arm on the
//! core, bit-deterministic across reps): **ZERO CONVERSIONS at every operating
//! point.** On the three most-marginal cifar100 failing rows (idx_5242 -0.079,
//! idx_2132 -0.137, idx_4597 -0.166 baseline frontier-worst), arms i1/i2/i4/
//! s2/s3/d4 all end `timeout` with the frontier 60-300 open. The bound moves
//! exactly as the 2x2 promised (best arm: -0.0662 vs off -0.0790 on idx_5242)
//! but the remaining gap at the wall is 5-25x the whole armed-vs-off delta.
//! Cost ledger: ~360-400 ms/certified trial, supergradient ~26% of ascent
//! time, and the throughput loss is DOMINATED by the fixed per-domain overhead
//! (`Collect::unst_rows` on the main pass + gradient), not by the trials —
//! s3 pays 1/3 of i2's trials and still loses 60 of off's 311 expansions on
//! idx_4597. So the SCHEDULE knobs cannot buy the cost back: stride-2 ~ties i2
//! on bound at slightly more expansions, and min_depth=4 is strictly worse
//! than i2 (the inheritance model predicted this: shallow accepted alphas are
//! what every descendant inherits). On tinyimagenet the lane is root-build
//! starved and arming HALVES the tree (71 -> 31 expansions on idx_2146,
//! worst -0.135 -> -0.170); gap=0.02 declines every domain on rows this deep
//! and is bit-identical to off. MOAT: 3 banked unsat re-proved under i2 AND
//! s2, 2 banked sat unchanged. VERDICT: keep dark on every shipped preset;
//! do not re-run this grid — the next lever must attack the fixed
//! per-expansion cost or the counting problem, not the slope quality.
//!
//! GATE. Default OFF and byte-identical with the key absent: no slots are
//! collected, no gate is written, `Collect::unst_rows` stays off, and the lane
//! runs exactly the passes it ran before.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::bab::TrunkSplit;
use super::engine::{DomainGates, GateVecs};
use super::root::RootGates;

/// A neuron this domain may move: `li` is the trunk relu layer, `idx` the
/// neuron index within that layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slot {
    /// Trunk relu layer index.
    pub li: usize,
    /// Neuron index within the layer (NOT the `unst` position).
    pub idx: usize,
}

/// One inherited alpha: `(layer, neuron, alpha)`. `u32` because a trunk layer
/// index and a neuron index both fit comfortably, and the frontier holds one
/// of these lists per open domain.
pub(crate) type AlphaCarry = (u32, u32, f64);

/// How many moved neurons a node hands to its children.
///
/// The step is a NORMALIZED supergradient move (`a + eta*g/gmax`), so the
/// neuron with the largest `|g|` moves by `eta` and everything else moves
/// strictly less, in proportion. Truncating to the largest `|delta|` therefore
/// keeps essentially all of the signal while bounding the frontier's memory to
/// `CARRY_CAP * 16 B` per open domain (~16 KB at 1024). A DROPPED entry is not
/// a soundness event: the child simply starts that neuron from the root's own
/// alpha, which is where it started before this module existed.
pub(crate) const CARRY_CAP: usize = 1024;

/// Upper bound on the ascent step size (accept => `eta * 1.5`, capped here).
pub(crate) const ETA_MAX: f64 = 1.0;

/// Lower bound on the ascent step size (reject => `eta * 0.5`, stop below).
/// A move smaller than this cannot change a `[0, 1]` slope enough to matter
/// against the certified pass's own scale.
pub(crate) const ETA_MIN: f64 = 0.01;

/// Compatibility preset cell for the master gate (`margin_row.domain_alpha`).
static ON_CELL: ny_levers::gate::BoolCell = ny_levers::gate::BoolCell::new();
/// Compatibility preset cell for the trial count (`margin_row.domain_alpha_iters`).
static ITERS_CELL: ny_levers::gate::U64Cell = ny_levers::gate::U64Cell::new();
/// Compatibility preset cell for the initial step (`margin_row.domain_alpha_eta`).
static ETA_CELL: ny_levers::gate::F64Cell = ny_levers::gate::F64Cell::new();
/// Typed preset cell for the depth floor (`margin_row.domain_alpha_min_depth`).
static MIN_DEPTH_CELL: ny_levers::gate::U64Cell = ny_levers::gate::U64Cell::new();
/// Typed preset cell for the depth stride (`margin_row.domain_alpha_stride`).
static STRIDE_CELL: ny_levers::gate::U64Cell = ny_levers::gate::U64Cell::new();
/// Typed preset cell for the gap window (`margin_row.domain_alpha_gap`).
static GAP_CELL: ny_levers::gate::F64Cell = ny_levers::gate::F64Cell::new();

/// The typed preset's answers for the whole `#margin-row-domain-alpha` family,
/// carried as one value so the CLI bridge writes every cell in one place and a
/// new knob cannot be added to the reader without being added to the bridge.
/// Every field is `None` when the yaml is silent, which resolves exactly as the
/// declaration defaults do.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Preset {
    /// `margin_row.domain_alpha` — the master gate.
    pub on: Option<bool>,
    /// `margin_row.domain_alpha_iters` — ascent trials per admitted domain.
    pub iters: Option<u64>,
    /// `margin_row.domain_alpha_eta` — initial normalized step.
    pub eta: Option<f64>,
    /// `margin_row.domain_alpha_min_depth` — skip the ascent above this depth.
    pub min_depth: Option<u64>,
    /// `margin_row.domain_alpha_stride` — pay the ascent every k-th depth.
    pub stride: Option<u64>,
    /// `margin_row.domain_alpha_gap` — pay only within this much of closure.
    pub gap: Option<f64>,
}

/// Record a compatibility caller's typed preset answers.
///
/// Scored VNN-COMP runs bypass these cells and carry a receipt-owned
/// `DomainAlphaRuntimePolicy`. A caller without that frozen set must invoke
/// this once before any margin-row lane starts. Absent keys leave every cell
/// empty, which resolves exactly as the declaration defaults do.
pub(crate) fn set_preset(p: Preset) {
    if let Some(on) = p.on {
        ON_CELL.set(on);
    }
    if let Some(iters) = p.iters {
        ITERS_CELL.set(iters);
    }
    if let Some(eta) = p.eta {
        ETA_CELL.set(eta);
    }
    if let Some(d) = p.min_depth {
        MIN_DEPTH_CELL.set(d);
    }
    if let Some(k) = p.stride {
        STRIDE_CELL.set(k);
    }
    if let Some(g) = p.gap {
        GAP_CELL.set(g);
    }
}

/// Compatibility resolver for the per-domain alpha gate.
///
/// Environment wins in both directions over the compatibility preset; absent
/// from both, the declaration's `false` default stands. Scored routes do not
/// call this function after freezing their runtime policy.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = ny_levers::gate::resolve_bool(
            &ny_levers::decls::margin_row::MARGIN_ROW_DOMAIN_ALPHA,
            &ON_CELL,
        );
        if on {
            // R9 engagement telemetry: an armed run announces itself once even
            // if no domain ever reaches the ascent, so an INERT arming is
            // detectable in one log. A null from a lever that never fired is
            // vacuous, and this campaign has banked several.
            eprintln!(
                "[dom-alpha] armed (iters={} eta={:.3} min_depth={} stride={} gap={} \
carry_cap={CARRY_CAP})",
                iters(),
                eta0(),
                min_depth(),
                stride(),
                gap(),
            );
        }
        on
    })
}

/// Compatibility resolver for ascent trials per expanded domain.
///
/// Each trial is ONE more certified pass on every expanded domain that has a
/// movable neuron — the whole cost model. This path and
/// `DomainAlphaRuntimePolicy::normalized` both clamp to `1..=8`, rather than
/// the declaration doing so, so an explicit `0` stays distinguishable from
/// absence in the receipt (the `beta.rs::iters` convention).
pub(crate) fn iters() -> usize {
    static I: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *I.get_or_init(|| {
        ny_levers::gate::resolve_u64(
            &ny_levers::decls::margin_row::MARGIN_ROW_DOMAIN_ALPHA_ITERS,
            &ITERS_CELL,
        )
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(2)
        .clamp(1, 8)
    })
}

/// Compatibility resolver for the initial normalized step size (default 0.25).
///
/// The step moves the largest-`|g|` neuron by exactly this much in `[0, 1]`
/// slope units, so 0.25 is a quarter of the whole legal range — the same
/// starting value `alpha_opt`'s root search uses, and for the same reason:
/// the certified scorer plus monotone accept absorbs an overshoot in one
/// rejected trial.
pub(crate) fn eta0() -> f64 {
    static E: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        ny_levers::gate::resolve_f64(
            &ny_levers::decls::margin_row::MARGIN_ROW_DOMAIN_ALPHA_ETA,
            &ETA_CELL,
        )
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(0.25)
        .clamp(ETA_MIN, ETA_MAX)
    })
}

/// Depth floor (default 0 = every depth pays).
///
/// THE SCHEDULE EXISTS BECAUSE THE COST AND THE GAIN HAVE DIFFERENT SHAPES.
/// The ascent costs [`iters`] certified passes at EVERY admitted domain,
/// uniformly in depth; the measured gain on the depth-indexed gating curve
/// GROWS with depth (+0.0017 at depth 1 to +0.0137 at depth 12 on idx_4597;
/// +0.0111 to +0.0495 on idx_1298). A schedule buys the deep part of that curve
/// without paying the shallow part, and the WARM START is what makes skipping
/// affordable: a child inherits its parent's accepted alpha for free
/// (0.98-1.12x pass-cost ratio at every width), so an unpaid node still carries
/// the tightening its ancestors bought.
///
/// Skipping is a TIGHTNESS choice and never a soundness one: an unpaid domain
/// publishes exactly the bound the disarmed lane would publish (or better, from
/// its inheritance), and the certified pass is still the only scorer.
pub(crate) fn min_depth() -> usize {
    static D: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *D.get_or_init(|| {
        ny_levers::gate::resolve_u64(
            &ny_levers::decls::margin_row::MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH,
            &MIN_DEPTH_CELL,
        )
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(0)
        .min(64)
    })
}

/// Depth stride (default 1 = every depth pays, clamped to `1..=16`).
///
/// `depth % stride == 0` admits, so `stride = 3` pays at depths 0, 3, 6, ...
/// and lets inheritance carry the accepted alpha across the two unpaid levels
/// between. This is the amortization arm: if the gain is carried by the WARM
/// START rather than by each node's own two trials, stride divides the cost by
/// `k` and keeps most of the curve; if each node's own trials are what move it,
/// stride divides the gain too. Those are distinguishable outcomes and that is
/// the point of the knob.
pub(crate) fn stride() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        ny_levers::gate::resolve_u64(
            &ny_levers::decls::margin_row::MARGIN_ROW_DOMAIN_ALPHA_STRIDE,
            &STRIDE_CELL,
        )
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(1)
        .clamp(1, 16)
    })
}

/// Closure-gap window (default 0.0, which reads as UNLIMITED — every domain is
/// admitted).
///
/// cifar100 fails by a COUNTING problem, not a bound-quality one: the frontier
/// goes 18 -> 41 -> 156 -> 297 -> 415 and children never close. What converts a
/// row is therefore not a better frontier-WORST but MORE DOMAINS CLOSING, and
/// the domains that can be made to close are the ones already near the closure
/// threshold (`b > 0`). Two certified passes spent on a domain at `-0.30` buy
/// ~+0.002 and close nothing; the same two spent at `-0.004` can close it
/// outright and remove it, and its children, from the frontier forever. This
/// gate admits only `b >= -gap`.
///
/// `0` (and any non-finite or negative value) means no window, i.e. the
/// unscheduled behaviour, so a silent preset is byte-identical to the arm the
/// mechanism was measured with.
pub(crate) fn gap() -> f64 {
    static G: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        ny_levers::gate::resolve_f64(
            &ny_levers::decls::margin_row::MARGIN_ROW_DOMAIN_ALPHA_GAP,
            &GAP_CELL,
        )
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(f64::INFINITY)
    })
}

/// Does the schedule admit this domain's ascent?
///
/// `depth` is the domain's split count (`trunk + heads`) and `b` its bound
/// BEFORE the ascent, so `-b` is its distance to closure in the production
/// (strict `b > 0`) predicate. Pure and total: this is the whole schedule, and
/// it is unit-tested rather than only observed through a 100 s run. The three
/// knobs are passed in from the caller's frozen
/// [`DomainAlphaRuntimePolicy`](super::DomainAlphaRuntimePolicy) — the lane
/// never reopens the process-wide `OnceLock` latches, which remain solely the
/// non-scored compatibility resolvers `from_layered_process` consumes once at
/// entry.
///
/// A non-finite `b` is REFUSED whenever a finite window is set: "how far is
/// this domain from closing" is not answerable for a NaN, and the ascent has
/// nothing to protect there. With no window (`gap` infinite, the default) the
/// bound is not consulted at all and the pre-schedule behaviour stands exactly.
pub(crate) fn schedule_admits_with(
    depth: usize,
    b: f64,
    min_depth: usize,
    stride: usize,
    gap: f64,
) -> bool {
    if depth < min_depth {
        return false;
    }
    if !depth.is_multiple_of(stride.max(1)) {
        return false;
    }
    if gap.is_finite() && !(b.is_finite() && b >= -gap) {
        return false;
    }
    true
}

/// The layer's effective gate vectors for this domain, materializing an
/// override from the root's when the domain has none yet.
///
/// `domain_gates` only inserts a layer when a split or a clip touched it, so
/// the first alpha write to an untouched layer has to clone the root's vectors
/// exactly as `clip_tighten_gates` does. `None` when the layer index is not a
/// trunk relu layer.
fn gate_mut<'a>(root: &RootGates, dom: &'a mut DomainGates, li: usize) -> Option<&'a mut GateVecs> {
    let layer = root.layers.get(li)?;
    Some(dom.layers.entry(li).or_insert_with(|| GateVecs {
        alpha: layer.alpha.clone(),
        s: layer.s.clone(),
        c: layer.c.clone(),
        ms: layer.ms.clone(),
    }))
}

/// This domain's `s` vector for layer `li` — the override where one exists,
/// the root's otherwise.
fn effective_s<'a>(root: &'a RootGates, dom: &'a DomainGates, li: usize) -> Option<&'a [f64]> {
    let root_s = root.layers.get(li)?.s.as_slice();
    Some(dom.layers.get(&li).map_or(root_s, |g| g.s.as_slice()))
}

/// The neurons this domain may move, in `(layer, neuron)` order.
///
/// ADMISSION IS `0 < s < 1`, which is precisely "this neuron carries a
/// relaxation TRIANGLE in this domain" and is therefore the exact complement
/// of every exact-line case: stable-active gives `(1, 1, 0)`, stable-inactive
/// `(0, 0, 0)`, a baked split `(1, 1, 0)` or `(0, 0, 0)`, and a neuron CLIP
/// stabilized is rebuilt through `gates_from_box` into one of those two. The
/// split positions are excluded explicitly on top of that, so a fresh split
/// whose gate has not been rebuilt cannot be loosened by an inherited value.
///
/// Restricted to `LayerGates::unst` because the gradient reads
/// `PassOut::unst_rows`, whose row order IS `unst` — a neuron outside that
/// list has no captured coefficient and so no direction.
pub(crate) fn movable(root: &RootGates, dom: &DomainGates, trunk: &[TrunkSplit]) -> Vec<Slot> {
    let mut out = Vec::new();
    for (li, layer) in root.layers.iter().enumerate() {
        if layer.unst.is_empty() {
            continue;
        }
        let Some(s) = effective_s(root, dom, li) else {
            continue;
        };
        for (pos, &idx) in layer.unst.iter().enumerate() {
            if trunk
                .iter()
                .any(|&(l2, p2, d)| l2 == li && p2 == pos && d != 0)
            {
                continue; // split-pinned: exact line, never relax it
            }
            let Some(&sv) = s.get(idx) else { continue };
            if !(sv.is_finite() && sv > 0.0 && sv < 1.0) {
                continue; // not a relaxation triangle here
            }
            out.push(Slot { li, idx });
        }
    }
    out
}

/// The domain's current alpha at each slot (the ascent's starting iterate).
pub(crate) fn read_alphas(root: &RootGates, dom: &DomainGates, slots: &[Slot]) -> Vec<f64> {
    slots
        .iter()
        .map(|sl| {
            dom.layers
                .get(&sl.li)
                .map_or_else(|| root.layers[sl.li].alpha[sl.idx], |g| g.alpha[sl.idx])
        })
        .collect()
}

/// Write `vals` into the domain's gates, clamped to `[0, 1]`, re-deriving the
/// error-contraction constant `ms = max(alpha, s)` at every touched neuron.
///
/// The clamp and the `ms` re-derivation are the module's two soundness
/// obligations and they live HERE, at the single write site, so no caller can
/// spell a write that skips either.
pub(crate) fn write_alphas(root: &RootGates, dom: &mut DomainGates, slots: &[Slot], vals: &[f64]) {
    for (sl, &a) in slots.iter().zip(vals) {
        let a = if a.is_finite() {
            a.clamp(0.0, 1.0)
        } else {
            continue;
        };
        let Some(gv) = gate_mut(root, dom, sl.li) else {
            continue;
        };
        let (Some(av), Some(&sv)) = (gv.alpha.get_mut(sl.idx), gv.s.get(sl.idx)) else {
            continue;
        };
        *av = a;
        if let Some(msv) = gv.ms.get_mut(sl.idx) {
            *msv = a.max(sv);
        }
    }
}

/// Apply an ancestor's accepted alpha to this domain, in place.
///
/// Only slots this domain still considers movable are written — an inherited
/// value for a neuron that has since been SPLIT (exact line) or clip-stabilized
/// is dropped, never applied over an exact line. `cur` is updated to match, so
/// the ascent's starting iterate is the warm-started one. Returns how many
/// entries were applied (R9: `warm=` in the telemetry).
pub(crate) fn apply_inherited(
    root: &RootGates,
    dom: &mut DomainGates,
    slots: &[Slot],
    cur: &mut [f64],
    inherited: &[AlphaCarry],
) -> usize {
    if inherited.is_empty() || slots.is_empty() {
        return 0;
    }
    let mut applied = 0usize;
    for &(li, idx, a) in inherited {
        let (li, idx) = (li as usize, idx as usize);
        // Slots are built in (layer, neuron) order, so this is a binary search.
        let Ok(k) = slots.binary_search_by(|sl| (sl.li, sl.idx).cmp(&(li, idx))) else {
            continue;
        };
        if !a.is_finite() {
            continue;
        }
        cur[k] = a.clamp(0.0, 1.0);
        applied += 1;
    }
    if applied > 0 {
        write_alphas(root, dom, slots, cur);
    }
    applied
}

/// This domain's effective per-layer alpha vectors — the input the linearized
/// walk needs (`alpha_opt::supergradient`).
pub(crate) fn effective_alpha(root: &RootGates, dom: &DomainGates) -> Vec<Vec<f64>> {
    root.layers
        .iter()
        .enumerate()
        .map(|(li, layer)| {
            dom.layers
                .get(&li)
                .map_or_else(|| layer.alpha.clone(), |g| g.alpha.clone())
        })
        .collect()
}

/// One projected, normalized supergradient step from `cur`.
///
/// `a' = clamp(a + eta * g / gmax, 0, 1)`, with `gmax` the largest `|g|` over
/// the MOVABLE slots only — so a huge gradient at a pinned neuron cannot
/// shrink everyone else's step. `None` when the direction is degenerate
/// (`gmax` zero or non-finite) or the projection leaves every slot unchanged,
/// which is the loop's stop condition.
///
/// Heuristic-only: nothing here feeds a verdict. Every proposal is re-scored
/// by the unchanged certified pass and kept only on a measured improvement.
pub(crate) fn propose(slots: &[Slot], cur: &[f64], g: &[Vec<f64>], eta: f64) -> Option<Vec<f64>> {
    if slots.len() != cur.len() || !(eta.is_finite() && eta > 0.0) {
        return None;
    }
    let at = |sl: &Slot| -> f64 {
        g.get(sl.li)
            .and_then(|v| v.get(sl.idx))
            .copied()
            .filter(|x| x.is_finite())
            .unwrap_or(0.0)
    };
    let gmax = slots.iter().map(|sl| at(sl).abs()).fold(0.0f64, f64::max);
    if !(gmax.is_finite() && gmax > 0.0) {
        return None;
    }
    let mut out = Vec::with_capacity(cur.len());
    let mut moved = false;
    for (sl, &a0) in slots.iter().zip(cur) {
        let a1 = (a0 + eta * at(sl) / gmax).clamp(0.0, 1.0);
        if !a1.is_finite() {
            return None;
        }
        moved |= a1 != a0;
        out.push(a1);
    }
    moved.then_some(out)
}

/// The sparse `(layer, neuron, alpha)` list this node hands to its children:
/// every movable slot whose accepted alpha differs from the ROOT's own value,
/// truncated to the [`CARRY_CAP`] largest deltas (see that constant).
///
/// Entries are re-sorted by `(layer, neuron)` so [`apply_inherited`]'s binary
/// search over `slots` stays valid.
pub(crate) fn carry(root: &RootGates, slots: &[Slot], cur: &[f64]) -> Vec<AlphaCarry> {
    let mut out: Vec<(f64, AlphaCarry)> = Vec::new();
    for (sl, &a) in slots.iter().zip(cur) {
        let Some(&base) = root.layers.get(sl.li).and_then(|l| l.alpha.get(sl.idx)) else {
            continue;
        };
        if a == base || !a.is_finite() {
            continue;
        }
        let (Ok(li), Ok(idx)) = (u32::try_from(sl.li), u32::try_from(sl.idx)) else {
            continue;
        };
        out.push(((a - base).abs(), (li, idx, a)));
    }
    if out.len() > CARRY_CAP {
        out.sort_by(|x, y| y.0.total_cmp(&x.0));
        out.truncate(CARRY_CAP);
    }
    let mut carried: Vec<AlphaCarry> = out.into_iter().map(|(_, e)| e).collect();
    carried.sort_by_key(|&(li, idx, _)| (li, idx));
    carried
}

/// Engagement telemetry (R9). Rate-limited so a scored run stays readable
/// while an inert arming is still visible in one log.
///
/// THE THREE OUTCOMES SEPARATE CLEANLY, which is the whole point:
/// * "never ran" — no `[dom-alpha] call=` line at all (only the `armed` line);
/// * "ran, no improvement" — `steps>0 accepted=0`, and `b0 == b1`;
/// * "ran and improved" — `accepted>0` with `b1 > b0`.
///
/// `movable=` prices the mechanism's reach (how many neurons the root froze
/// that this domain could move) and `warm=` proves the parent's accepted alpha
/// actually arrived — the two ways this could read as armed while doing
/// nothing.
pub(crate) fn report(
    depth: usize,
    movable: usize,
    warm: usize,
    steps: usize,
    accepted: usize,
    b0: f64,
    b1: f64,
    eta: f64,
) {
    static N: AtomicUsize = AtomicUsize::new(0);
    static ACC: AtomicUsize = AtomicUsize::new(0);
    static WARM: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    ACC.fetch_add(accepted, Ordering::Relaxed);
    WARM.fetch_add(warm, Ordering::Relaxed);
    if n < 8 || n.is_multiple_of(128) {
        eprintln!(
            "[dom-alpha] call={n} depth={depth} movable={movable} warm={warm} steps={steps} \
accepted={accepted} b0={b0:+.6} b1={b1:+.6} delta={:+.3e} eta={eta:.3} accepted_total={} \
warm_total={} skipped={} grad_ms={} trial_ms={} trials={}",
            b1 - b0,
            ACC.load(Ordering::Relaxed),
            WARM.load(Ordering::Relaxed),
            SKIPPED.load(Ordering::Relaxed),
            GRAD_US.load(Ordering::Relaxed) / 1000,
            TRIAL_US.load(Ordering::Relaxed) / 1000,
            TRIALS.load(Ordering::Relaxed),
        );
    }
}

/// Domains the schedule declined (`schedule_admits` false) — the saving side of
/// the ledger, in the same units as the `call=` count above.
static SKIPPED: AtomicUsize = AtomicUsize::new(0);
/// Ascent trials actually run (certified passes), across all workers.
static TRIALS: AtomicUsize = AtomicUsize::new(0);
/// Microseconds inside `alpha_opt::supergradient` — the ascent's FIXED
/// overhead, charged once per admitted domain plus once per non-final accept.
static GRAD_US: AtomicU64 = AtomicU64::new(0);
/// Microseconds inside the certified trial passes — the ascent's MARGINAL cost,
/// the part that scales with [`iters`].
static TRIAL_US: AtomicU64 = AtomicU64::new(0);

/// Count a domain the schedule declined.
pub(crate) fn note_skipped() {
    SKIPPED.fetch_add(1, Ordering::Relaxed);
}

/// Charge a supergradient evaluation (fixed overhead).
pub(crate) fn charge_grad(us: u64) {
    GRAD_US.fetch_add(us, Ordering::Relaxed);
}

/// Charge one certified trial pass (marginal cost).
pub(crate) fn charge_trial(us: u64) {
    TRIAL_US.fetch_add(us, Ordering::Relaxed);
    TRIALS.fetch_add(1, Ordering::Relaxed);
}

/// One end-of-lane cost line, so a run's WHOLE ledger is readable without
/// waiting for a rate-limited sample: how many domains paid, how many the
/// schedule declined, and how the paid time split between the FIXED
/// supergradient overhead and the MARGINAL trial passes. That split is what
/// says whether `iters` is a linear knob or whether there is overhead to
/// amortize.
pub(crate) fn cost_summary() {
    let trials = TRIALS.load(Ordering::Relaxed);
    let skipped = SKIPPED.load(Ordering::Relaxed);
    if trials == 0 && skipped == 0 {
        return;
    }
    let trial_ms = TRIAL_US.load(Ordering::Relaxed) as f64 / 1000.0;
    let grad_ms = GRAD_US.load(Ordering::Relaxed) as f64 / 1000.0;
    let per_trial_ms = if trials > 0 { trial_ms / trials as f64 } else { 0.0 };
    let total_ms = trial_ms + grad_ms;
    let fixed_frac = if total_ms > 0.0 { grad_ms / total_ms } else { 0.0 };
    eprintln!(
        "[dom-alpha] COST skipped={skipped} trials={trials} trial_ms={trial_ms:.0} \
grad_ms={grad_ms:.0} per_trial_ms={per_trial_ms:.1} fixed_frac={fixed_frac:.3}"
    );
}

/// Refusal telemetry (R9: every refusal path prints). Rate-limited globally;
/// the site tag makes a systematically-refusing lane visible in one log.
pub(crate) fn refuse(site: &str) {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 8 || n.is_multiple_of(256) {
        eprintln!("[dom-alpha] refuse={site} n={n}");
    }
}
