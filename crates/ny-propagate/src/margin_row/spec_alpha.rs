// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#spec-alpha` — the per-START-NODE (spec) alpha axis at the root direct pass.
//!
//! # What this is
//!
//! alpha-beta-CROWN keeps `alpha[relu][start_node]` of shape
//! `[2, spec, batch, *relu_shape]`
//! (`references/alpha-beta-CROWN/auto_LiRPA/auto_LiRPA/operators/relu.py:155`,
//! `backward_bound.py:927-944`): every output neuron of every start node gets
//! its OWN lower-relaxation slope on every upstream ReLU. ny has one shared
//! `(layer, neuron)` alpha, so every adversarial class is bounded with the same
//! relaxation even when their optimal relaxations disagree.
//!
//! This module adds exactly the axis abc keeps when it throws the rest away:
//! the FINAL start node's spec axis — in ny's margin-row lane, the adversarial
//! class. (abc's `share_alphas` collapses the INTERMEDIATE start nodes' spec
//! axis to 1, `backward_bound.py:927-929`, but `continue`s past that collapse
//! for the final node, `:918-925`; and `alpha_drop_unused`,
//! `complete_verifier/state/alpha.py`, discards every intermediate start node
//! before BaB.)
//!
//! # The mechanism, in one paragraph
//!
//! After the root direct pass has produced `dj[k]` per adversarial class, run
//! ONE certified collect pass at the PUBLISHED alpha to recover each selected
//! class's own supergradient (the caller's direct pass is `run_seamed`, which
//! captures none), then ONE additional certified backward pass whose COLUMNS
//! are `(hardest class k, candidate alpha e)` pairs — the candidates become
//! pass WIDTH, not pass COUNT — delivered through the engine's existing
//! [`Exceptions`](super::engine::Exceptions) per-`(row, neuron)` gate
//! replacement, and fold the result into `dj[k]` by ELEMENT-WISE MAX.
//!
//! The PRIMARY candidate for class `k` is
//! `clamp(alpha_incumbent + eta * sign(g_k), 0, 1)` — a per-class step from the
//! point the root actually published. The closed-form ladder is a cheap
//! SECONDARY family and is global-uniform, so its per-class content is
//! selection only. `spec_alpha_rounds >= 2` keeps stepping.
//!
//! Nothing is mutated. Not `LayerGates::alpha`, not `ms`, not `s`/`c`/`l`/`u`,
//! not `clip_rows`, not `unst`, not the y-pack, not the head gates, not the
//! tree's domain gates. There is no commit, no rollback state, and no guard to
//! write, because `dj[k] = dj[k].max(spec[k])` IS the guard.
//!
//! # Soundness (three sentences, zero new certified arithmetic)
//!
//! 1. **Per-entry validity ships already.** `alpha*z <= relu(z)` holds
//!    pointwise on all of R for any `alpha in [0,1]`, box-independently. The one
//!    alpha-coupled certified quantity is `ms = max(alpha, s)`, the ATTAINED
//!    Lipschitz constant of the gate map `v -> v+*alpha + v-*s` (derivation at
//!    `root.rs`'s `gate_gerr_out`; obligation stated on
//!    [`LayerGates::alpha`](super::root::LayerGates::alpha)). The exception path
//!    RE-DERIVES it from the replacement slopes itself —
//!    `let m2 = x.a2.abs().max(x.s2.abs());` in `engine.rs`'s
//!    `run_collect_impl` — feeding an error update formula-identical to the
//!    dense transform's. This module never hand-writes `ms`, so there is no
//!    obligation for it to violate.
//! 2. **The direct pass is COLUMN-SEPARABLE**, so a pass at width `R` is `R`
//!    already-certified passes fused. Bias and bias-error accumulate per column
//!    and read only `c` (never `alpha`/`ms`); the gate transform is elementwise
//!    over `(neuron, column)` with the certified error carry `cur.e` already an
//!    `(n, r)` matrix; the rounding charges are functions of the layer width,
//!    never of `r`; `concretize` is indexed per column. The ONE cross-column
//!    reduction in the engine is the branching-shortlist `unst_abs` sum, which
//!    this module never requests. With `s2 = layer.s[j]` and `c2 = layer.c[j]`
//!    the exception's bias correction `vn * (c2 - c[neuron])` is EXACTLY `0.0`,
//!    so `b` is bitwise what a rank-1 pass under the candidate alpha produces;
//!    the only residue is one extra `slack16` per entry on `eb`, which INFLATES
//!    the bias error, i.e. LOWERS the certified bound. Safe direction, by
//!    inspection. ([`tests`](super::tests) `spec_alpha_replicated_pass_is_column_exact`
//!    establishes this by execution.)
//! 3. **The published number is an element-wise max of independently certified
//!    bounds**: `dj[k] = max(m1, m2v, direct, spec_1, spec_2, ...)`. Each term
//!    is a valid lower bound on class `k`'s margin, so `dj_armed[k] >=
//!    dj_dark[k]` identically, for every `k`, for every candidate, however bad.
//!    A garbage candidate costs wall and nothing else.
//!
//! ## Two boundaries held by construction, not by discipline
//!
//! * **The FORWARD lane is never entered** — by construction of THIS module,
//!   which is backward-only and never touches `apply_gates`/`walk_ops`. It used
//!   to be claimed that the forward lane was closed to this axis in principle,
//!   because `gate_gerr_out`'s derivation "requires alpha and s UNIFORM over the
//!   coefficient index". That was wrong: uniformity is what lets a single factor
//!   pull out of the aggregate, not what makes the bound true, and
//!   `gate_gerr_out_nonuniform` carries the generalized charge
//!   `max_i max(a_ji, s_ji)` for slopes that vary over that index — a valid
//!   ERROR carry, still contracting because every slope is in `[0, 1]`. That
//!   generalizes the error functional ONLY. The margin-row tableau's columns
//!   are the network's INPUT coordinates plus a bias (`naug = n_in + 1`), not
//!   start nodes, so
//!   a non-uniform alpha over them is a per-input-coordinate alpha, and whether
//!   the RELAXATION stays a valid enclosure under such a slope is OPEN. Do NOT
//!   arm it on the forward walk. `#rebake` DOES touch the forward lane — and
//!   it runs strictly BEFORE [`SpecAlphaPolicy`] is installed on the root
//!   (`margin_row/mod.rs`), so `rebake::certified_root_bound` and every
//!   `#alpha-opt` guard evaluation observe `None`.
//! * **Tree monotonicity absorbs a raised root.** A class the axis closes is
//!   removed from `tree_classes` — strictly less work, certified closed by the
//!   unchanged `closed(mode, dj)`. Children already clamp up to their parent.
//!
//! # Why `Exceptions` and not a rank-2 `GateVecs`
//!
//! Widening `GateVecs::alpha` to rank 2 forces the `(aj, sj, mj)` lookup INSIDE
//! the hot column loop and propagates to `beta`, `beta_pc`,
//! `clip_tighten_gates` (which would OVERWRITE a per-column axis by rebuilding
//! from `gates_from_box`), `domain_gates_split_only`, `#domain-alpha`,
//! `#rebake` and the GPU quad builder. `Exceptions` delivers identical
//! expressiveness with ZERO new certified arithmetic.
//!
//! # Reading the telemetry
//!
//! * no `[spec-alpha] armed:` line  => **VOID** (the binary lacks the lever, or
//!   the yaml route is dark). A null from a run with no `armed:` line is not a
//!   negative.
//! * `armed:` + `apply: ... entries=0` => **VOID by config** (nothing to move).
//! * `armed:` + `entries>0` + `classes_improved=0` => **REAL NEGATIVE**.
//! * any `[spec-alpha] refuse:` line => **VOID for that named cause**.
//!
//! Two further rules, because the mechanism has TWO candidate families and a
//! null about one is not a null about the other:
//!
//! * `refuse: incumbent pass:` => the per-class gradient family never ran and
//!   only the GLOBAL UNIFORM ladder was measured. **VOID for the axis**, and
//!   the ladder is the already-measured `#interm-alpha` family.
//! * `done: ... incumbent_grad_improvements=0` with a positive `delta` => the
//!   yield came from the ladder, not from the per-start-node axis. The field
//!   counts every NON-LADDER winner (`IncumbentGrad` or `RoundGrad`), so it
//!   stays an honest family attribution at `rounds >= 2`; read it from the
//!   `rounds: 1` arm when you want the families separated cleanly.
//! * `done: ... distinct_winners=1` => every improving class ended on the SAME
//!   alpha, so a rank-1 "best global candidate" would have done just as well.
//!   That is a real negative ABOUT THE AXIS even when `delta > 0` — it is the
//!   `cols=8 delivers cols=1` shape, and it is the number to read first.

use std::time::Instant;

use ny_core::Result;

use super::bounds::{margin_seed, per_class_direct, HeadGates, MarginBatch, YBox};
use super::engine::{BackwardEngine, Collect, Exc, Exceptions, LaneDir};
use super::net::TwinNet;
use super::root::RootGates;

/// `#spec-alpha` policy, read at exactly ONE site (`bab::root_eval_impl`).
///
/// Carries the deadline because `root_eval` has no deadline parameter and the
/// root driver (`margin_row/mod.rs`) is the only place one is in scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecAlphaPolicy {
    /// `W`: how many of the HARDEST failing classes receive candidates.
    pub width: usize,
    /// `E`: candidates per selected class.
    pub cands: usize,
    /// `1` = closed-form ladder only; `>= 2` = additional gradient rounds.
    pub rounds: usize,
    /// Hard cap on committed exception entries. `0` refuses and says so.
    pub entries: usize,
    /// Instance deadline, if any.
    pub deadline: Option<Instant>,
}

impl SpecAlphaPolicy {
    /// Build from the frozen root-alpha policy. `None` when the key is dark —
    /// which is what makes the whole mechanism one `Option` discriminant.
    pub(crate) fn from_policy(
        policy: super::RootAlphaRuntimePolicy,
        deadline: Option<Instant>,
    ) -> Option<Self> {
        policy.spec_alpha_enabled.then_some(Self {
            width: policy.spec_alpha_width,
            cands: policy.spec_alpha_cands,
            rounds: policy.spec_alpha_rounds,
            entries: policy.spec_alpha_entries,
            deadline,
        })
    }
}

/// What one `#spec-alpha` refinement did. Evidence only; never feeds a bound.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpecAlphaReport {
    /// Rounds that actually ran a certified pass.
    pub rounds_run: usize,
    /// Certified backward passes issued.
    pub passes: usize,
    /// Exception entries committed across all rounds.
    pub entries: usize,
    /// Classes given candidates.
    pub classes_selected: usize,
    /// Selected classes for which the INCUMBENT supergradient resolved, i.e.
    /// for which the per-start-node axis had a per-class direction at all.
    ///
    /// `0` with `classes_selected > 0` means the mechanism degraded to the
    /// global-uniform ladder — a VOID for the axis, not a negative about it.
    pub incumbent_grads_ok: usize,
    /// DISTINCT classes whose `dj` strictly improved, across ALL rounds.
    ///
    /// A total, not a per-round maximum: round 1 improving class A and round 2
    /// improving class B is `2`.
    pub classes_improved: usize,
    /// Distinct winning ALPHAS among the classes that improved — i.e. how many
    /// different relaxations the per-start-node axis actually delivered.
    ///
    /// `1` means every improving class ended on the SAME alpha, so a rank-1
    /// "best global candidate" would have done just as well and the axis
    /// contributed selection only. This is the `cols=8 delivers cols=1`
    /// detector, and it is the number to read before `delta`.
    pub distinct_winners: usize,
    /// Improving classes whose winner came from the INCUMBENT-gradient family
    /// (`CandOrigin::IncumbentGrad`) rather than the global-uniform ladder.
    ///
    /// Zero here with a positive `delta` means the yield came from the already
    /// measured `#interm-alpha` ladder, not from this axis.
    pub incumbent_grad_improvements: usize,
    /// Classes the axis moved from failing to closed — a DELTA against the
    /// count already closed when `refine` was entered.
    ///
    /// The distinction is load-bearing: `fail` is computed BEFORE the root
    /// direct pass folds its own bounds into `dj`, so a plain count over
    /// `fail` credits this axis with every class the UNMODIFIED direct pass
    /// closed.
    pub closed_extra: usize,
    /// Best single-class gain.
    pub best_gain: f64,
    /// `min dj` over the failing set, before and after.
    pub root_before: f64,
    /// See [`Self::root_before`].
    pub root_after: f64,
    /// Whole candidates dropped by the entry cap.
    pub truncated: usize,
}

/// The candidate ladder: the five closed forms `start_node_gates` already
/// enumerates (`backward_interm.rs`), REORDERED most-distinct-first so a
/// truncated `cands` keeps the informative ones.
///
/// Arguments are `(alpha_j, s_j)`. Every value is clamped to `[0, 1]` by the
/// caller; that clamp is LOAD-BEARING, not tidiness — `repair_upper_lines` can
/// push `s` to the boundary in outward mode, and then `midpoint(s, 1)` would
/// exceed 1.
pub(crate) const SPEC_ALPHA_LADDER: [fn(f64, f64) -> f64; 5] = [
    // The vertex the binary area heuristic rejected.
    |a, _| if a >= 0.5 { 0.0 } else { 1.0 },
    // Parallel to the certified upper chord.
    |_, s| s,
    // Box midpoint.
    |_, _| 0.5,
    // Between the chord-parallel line and fully-active.
    |_, s| s.midpoint(1.0),
    // Between fully-inactive and the chord-parallel line.
    |_, s| s * 0.5,
];

/// Step sizes for the gradient candidates, MOST-DISTINCT-FIRST so a truncated
/// `cands` keeps the informative ones. `1.0` reaches the vertex; the rest are
/// interior. Same hard `[0, 1]` projection abc uses (`relu.py:433-435`,
/// `:613-636`) — there is deliberately NO reparameterisation, because abc gains
/// its +0.163 with the same clamp.
///
/// EIGHT entries, matching the `1..=8` clamp the reader applies to
/// `margin_row.spec_alpha_cands`. It was three, which silently delivered
/// `cands <= 3` on every gradient candidate while the declaration advertised
/// eight — the `cols=8 delivers cols=1` shape this campaign keeps finding. The
/// LADDER's cardinality (5) still caps the ladder family; that asymmetry is
/// stated in the key's declaration rather than left to be discovered.
pub(crate) const SPEC_ALPHA_ETA: [f64; 8] = [1.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.0625, 0.031_25];

/// Absolute ceiling on committed entries, regardless of the typed key.
///
/// `margin_row/` consults NO memory governor (`crown_memory` has zero callers
/// in this module tree) and competition runs with no cgroup, so this constant
/// and [`SpecAlphaPolicy::entries`] are the only bounds that exist.
pub(crate) const SPEC_ALPHA_MAX_ENTRIES: usize = 1 << 22;

/// Where a candidate came from. Carried into the telemetry because the two
/// families answer DIFFERENT questions and a null that cannot tell them apart
/// is uninterpretable.
///
/// * [`CandOrigin::Ladder`] values are GLOBAL UNIFORM alphas — the same move
///   list for every class. Their per-class content is SELECTION ONLY, and that
///   family is the already-measured `#interm-alpha` ladder (+0.0025 total).
///   A run in which only ladder candidates ever win has NOT exercised the
///   per-start-node axis.
/// * [`CandOrigin::IncumbentGrad`] is the axis itself:
///   `clamp(alpha_incumbent + eta * sign(g_k), 0, 1)` with `g_k` the class's
///   OWN supergradient at the PUBLISHED alpha.
/// * [`CandOrigin::RoundGrad`] is a further step from a previous round's best
///   candidate for that class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CandOrigin {
    /// Ladder index into [`SPEC_ALPHA_LADDER`].
    Ladder(usize),
    /// Step index into [`SPEC_ALPHA_ETA`], based at the incumbent alpha.
    IncumbentGrad(usize),
    /// Step index into [`SPEC_ALPHA_ETA`], based at a previous round's winner.
    RoundGrad(usize),
}

/// One candidate: SPARSE, because a full-width alpha per `(class, candidate)`
/// is the only object in this design that could reach tens of MB. Materialised
/// to full width only for the single best replica per class, and only when a
/// gradient round needs it.
pub(crate) struct Cand {
    /// Index into `fail`, i.e. the adversarial-class SLOT. Built in the same
    /// scope as `fail_classes` in the same call, so there is no stored class
    /// list that can go stale against a different fail set.
    pub(crate) class_pos: usize,
    /// This candidate's column in the refine pass.
    ///
    /// INVARIANT, asserted by [`build_exceptions`]: `cands[i].col == i`. The
    /// fold reads `spec[c.col]` and writes `dj[fail[c.class_pos]]`, and the
    /// replicated seed is built from `cands[i]` in order — so a PERMUTATION of
    /// columns folds one class's certified bound into another class's `dj`,
    /// which can close a class that was never proved. That is a false `unsat`,
    /// and it is the only way this module could produce one.
    pub(crate) col: usize,
    /// Which family produced this candidate.
    pub(crate) origin: CandOrigin,
    /// `(layer, neuron, alpha)`, neurons ascending within a layer.
    pub(crate) moves: Vec<(usize, usize, f64)>,
}

/// FNV-1a over a candidate's move list. Evidence only — it names WHICH alpha a
/// class ended on, so `distinct_winners` can say whether the per-class axis
/// delivered anything a single shared candidate would not.
fn fingerprint(moves: &[(usize, usize, f64)]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        for byte in v.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for &(li, j, a) in moves {
        mix(li as u64);
        mix(j as u64);
        mix(a.to_bits());
    }
    h
}

impl Cand {
    /// Materialise this candidate's full-width alpha (root alpha + moves).
    pub(crate) fn full_alpha(&self, root: &RootGates) -> Vec<Vec<f64>> {
        let mut a: Vec<Vec<f64>> = root.layers.iter().map(|l| l.alpha.clone()).collect();
        for &(li, j, v) in &self.moves {
            a[li][j] = v;
        }
        a
    }
}

/// Build the candidate ladder for one class slot.
pub(crate) fn ladder_moves(root: &RootGates, e: usize) -> Vec<(usize, usize, f64)> {
    let rule = SPEC_ALPHA_LADDER[e % SPEC_ALPHA_LADDER.len()];
    let mut moves = Vec::new();
    for (li, layer) in root.layers.iter().enumerate() {
        for &j in &layer.unst {
            let a = rule(layer.alpha[j], layer.s[j]).clamp(0.0, 1.0);
            // An identity exception spends a `slack16` on the bias-error lane
            // for a guaranteed zero. Skip it.
            if a != layer.alpha[j] {
                moves.push((li, j, a));
            }
        }
    }
    moves
}

/// Build a gradient-step candidate from a base alpha and a per-column
/// supergradient.
pub(crate) fn gradient_moves(
    root: &RootGates,
    base: &[Vec<f64>],
    grad: &[Vec<f64>],
    eta: f64,
) -> Vec<(usize, usize, f64)> {
    let mut moves = Vec::new();
    for (li, layer) in root.layers.iter().enumerate() {
        for &j in &layer.unst {
            let g = grad[li][j];
            // NOT `f64::signum`: `0.0f64.signum() == 1.0`, which would move
            // every neuron with NO support in this column — precisely the
            // neurons that must not move.
            let dir = if g > 0.0 {
                1.0
            } else if g < 0.0 {
                -1.0
            } else {
                0.0
            };
            // `dir == 0.0` means THIS column has no support at the neuron, not
            // that the base sits at the root. Skipping the push used to drop the
            // base's own displacement there, because an exception the engine
            // never receives falls back to `layer.alpha[j]` — so a round-2
            // candidate was "root (+) gradient moves from base", not "base
            // stepped by its own supergradient" as the doc claims. Emit the
            // unmoved base instead; it is still some alpha in [0, 1], so the
            // max fold guards it exactly as before.
            let a = if dir == 0.0 {
                base[li][j]
            } else {
                (base[li][j] + eta * dir).clamp(0.0, 1.0)
            };
            if a != layer.alpha[j] {
                moves.push((li, j, a));
            }
        }
    }
    moves
}

/// Assign columns and apply the entry cap, dropping WHOLE candidates from the
/// end (a half-applied ladder value is a different, undocumented relaxation).
/// Returns `(candidates, entries, truncated)`.
pub(crate) fn assign_columns(
    raw: Vec<(usize, CandOrigin, Vec<(usize, usize, f64)>)>,
    cap: usize,
) -> (Vec<Cand>, usize, usize) {
    let mut out = Vec::with_capacity(raw.len());
    let mut entries = 0usize;
    let mut truncated = 0usize;
    let mut col = 0usize;
    for (class_pos, origin, moves) in raw {
        if entries + moves.len() > cap {
            truncated += 1;
            continue;
        }
        entries += moves.len();
        // `col` counts ADMITTED candidates, so the emitted columns are dense
        // and ascending: `out[i].col == i`. `build_exceptions` re-asserts that
        // rather than trusting it — see `Cand::col`.
        out.push(Cand {
            class_pos,
            col,
            origin,
            moves,
        });
        col += 1;
    }
    (out, entries, truncated)
}

/// Build the `Exceptions` set for a candidate list, FAILING CLOSED on anything
/// the engine would not survive.
///
/// Three of these four checks guard a PANIC rather than a wrong answer: the
/// uniform exception arm indexes `cur.l[[x.neuron, x.row]]` with no validation
/// of its own (only the domain-stacked arm validates shapes), so an
/// out-of-range row or neuron aborts the process inside a verdict pass. The
/// fourth — duplicate `(row, neuron)` — would double-apply a gate and charge
/// the bias-error lane twice from a stale snapshot.
///
/// Returns `(exceptions, entries, max_row)`.
pub(crate) fn build_exceptions(
    root: &RootGates,
    cands: &[Cand],
    cols: usize,
) -> Result<(Exceptions, usize, usize)> {
    use ny_core::NyError;
    let mut exc = Exceptions::default();
    let mut seen: std::collections::BTreeSet<(usize, usize, usize)> =
        std::collections::BTreeSet::new();
    let mut entries = 0usize;
    let mut max_row = 0usize;
    for (i, c) in cands.iter().enumerate() {
        if c.col >= cols {
            return Err(NyError::InvalidSpec(format!(
                "margin_row #spec-alpha: candidate column {} out of range {cols}",
                c.col
            )));
        }
        // THE COLUMN->CLASS INVARIANT. `run_round` builds the replicated seed
        // as `replicated[i] = fail_classes[cands[i].class_pos]` and folds
        // `spec[c.col]` into `dj[fail[c.class_pos]]`. Those agree ONLY if
        // `cands[i].col == i`. The range check above and the duplicate check
        // below both pass on a PERMUTATION, and a permutation folds class A's
        // certified bound into class B's `dj` — closing a class that was never
        // proved. Refuse here, where the exception set is built, so no
        // rearrangement of `assign_columns` can reintroduce it silently.
        if c.col != i {
            return Err(NyError::InvalidSpec(format!(
                "margin_row #spec-alpha: candidate {i} carries column {} — columns must be \
                 dense and ascending or a class's bound is folded into another class",
                c.col
            )));
        }
        for &(li, j, a) in &c.moves {
            let Some(layer) = root.layers.get(li) else {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row #spec-alpha: candidate layer {li} out of range {}",
                    root.layers.len()
                )));
            };
            // Only UNSTABLE neurons may move: stable and split-baked neurons
            // keep their exact fixed lines, and a fractional lower line on a
            // stable-active neuron is not the relaxation its certificate
            // describes.
            if !layer.unst.contains(&j) {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row #spec-alpha: neuron {j} of layer {li} is not unstable"
                )));
            }
            if !(0.0..=1.0).contains(&a) || !a.is_finite() {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row #spec-alpha: candidate alpha {a} outside [0, 1]"
                )));
            }
            if !seen.insert((li, c.col, j)) {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row #spec-alpha: duplicate exception at (layer {li}, row {}, neuron {j})",
                    c.col
                )));
            }
            exc.by_layer.entry(li).or_default().push(Exc {
                row: c.col,
                neuron: j,
                a2: a,
                // `s`/`c` are functions of `(l, u)` ONLY and `(l, u)` are frozen
                // here, so the certified chord is emitted VERBATIM. This is what
                // makes the exception's bias correction `vn * (c2 - c[neuron])`
                // exactly `0.0` and the pass bitwise equal to a rank-1 pass
                // under the candidate alpha, save the one-directional `slack16`
                // on the bias-error lane.
                s2: layer.s[j],
                c2: layer.c[j],
            });
            entries += 1;
            max_row = max_row.max(c.col);
        }
    }
    Ok((exc, entries, max_row))
}

/// Per-class state carried across rounds: the winning alpha's fingerprint and
/// which family produced it. Evidence only.
type Winners = std::collections::BTreeMap<usize, (u64, CandOrigin)>;

/// The INCUMBENT: the published root alpha, one certified collect pass over
/// every failing class at that alpha, and each selected class's OWN
/// supergradient read out of that pass.
///
/// This pass exists because the armed root direct pass is `run_seamed`
/// (`bab::root_eval_impl`), which takes no [`Collect`], so `unst_rows` for the
/// incumbent exists NOWHERE else — and without it the axis can only hill-climb
/// from a deliberately perturbed global-uniform ladder value, never from the
/// published point. Running it HERE rather than widening the caller's pass
/// keeps the dark path untouched and keeps an A/B attributable: the dark arm's
/// `direct[fi]` stays bitwise what it was.
struct Incumbent {
    /// The published per-`(layer, neuron)` alpha.
    alpha: Vec<Vec<f64>>,
    /// Width-`fail.len()` pass with `unst_rows` captured; column `class_pos`
    /// is `fail_classes[class_pos]`'s own objective column.
    pass: super::engine::PassOut,
    /// `class_pos -> supergradient of column `class_pos` at `alpha`.
    grads: std::collections::BTreeMap<usize, Vec<Vec<f64>>>,
}

/// Build [`Incumbent`]. One extra certified pass, no exceptions, nothing
/// mutated; the result feeds only candidate CONSTRUCTION, never a bound.
fn incumbent_grads(
    eng: &BackwardEngine<'_>,
    net: &TwinNet,
    t: usize,
    gates: &HeadGates,
    ybox: &YBox,
    fail_classes: &[usize],
    order: &[usize],
) -> Result<Incumbent> {
    let root = eng.root;
    let mb = MarginBatch::new(net, t, fail_classes)?;
    let ms = margin_seed(&mb, gates, ybox, root.mode);
    let collect = Collect {
        unst_abs: false,
        rows: None,
        unst_rows: true,
    };
    let pass = eng.run_collect(&ms.seed, None, LaneDir::Lower, None, collect)?;
    let alpha: Vec<Vec<f64>> = root.layers.iter().map(|l| l.alpha.clone()).collect();
    let mut grads = std::collections::BTreeMap::new();
    for &class_pos in order {
        // `supergradient` is reused UNCHANGED (`alpha_opt`): it reads
        // `pass.unst_rows` for this column, rebuilds the concretize argmin per
        // column and returns `g[li][j] = vp * U`. `mean_supergradient`
        // computes exactly this quantity and then destroys it by averaging;
        // this never averages. Direction-only — nothing verdict-bearing reads
        // it — so a `None` costs candidates, never correctness.
        if let Some(g) = super::alpha_opt::supergradient(net, root, &alpha, &pass, class_pos) {
            grads.insert(class_pos, g);
        }
    }
    Ok(Incumbent { alpha, pass, grads })
}

/// Run one certified refine pass over `cands` and fold the result into `dj` by
/// element-wise max. Returns the completed `PassOut` (for a following gradient
/// round) plus the per-column bounds.
#[allow(clippy::too_many_arguments)]
fn run_round(
    eng: &BackwardEngine<'_>,
    net: &TwinNet,
    t: usize,
    round: usize,
    cands: &[Cand],
    entries: usize,
    truncated: usize,
    fail: &[usize],
    fail_classes: &[usize],
    gates: &HeadGates,
    ybox: &YBox,
    want_grad: bool,
    dj: &mut [f64],
    improved_classes: &mut std::collections::BTreeSet<usize>,
    winners: &mut Winners,
    report: &mut SpecAlphaReport,
) -> Result<Option<(super::engine::PassOut, Vec<f64>)>> {
    let root = eng.root;
    let mode = root.mode;
    let r = cands.len();
    // COLUMN REPLICATION. `MarginBatch::new` builds each row independently from
    // `adv[r]` with no dedup and no uniqueness check, and `margin_seed` loops
    // per row independently, so a replicated class list yields bit-identical
    // duplicate columns. That is what makes candidates pass WIDTH rather than
    // pass COUNT.
    let replicated: Vec<usize> = cands.iter().map(|c| fail_classes[c.class_pos]).collect();
    let mbr = MarginBatch::new(net, t, &replicated)?;
    let msr = margin_seed(&mbr, gates, ybox, mode);

    let (exc, built_entries, max_row) = build_exceptions(root, cands, r)?;
    debug_assert_eq!(built_entries, entries);
    // The one observable printed from INSIDE the builder, at the read site in
    // the certified pass. It separates "the resolver said yes" from "the
    // exception set reached the engine".
    let grad_cols = cands
        .iter()
        .filter(|c| !matches!(c.origin, CandOrigin::Ladder(_)))
        .count();
    eprintln!(
        "[spec-alpha] apply: round={round} cols={r} grad_cols={grad_cols} ladder_cols={} \
entries={built_entries} layers={} max_row={max_row} truncated={truncated}",
        r - grad_cols,
        exc.by_layer.len()
    );

    let collect = Collect {
        unst_abs: false,
        rows: None,
        unst_rows: want_grad,
    };
    let pass = eng.run_collect(&msr.seed, None, LaneDir::Lower, Some(&exc), collect)?;
    let spec = per_class_direct(eng, &pass, &msr, 0..r);

    report.passes += 1;
    report.rounds_run += 1;
    report.entries += entries;
    report.truncated += truncated;

    for c in cands {
        let v = spec[c.col];
        if !v.is_finite() {
            eprintln!(
                "[spec-alpha] refuse: non-finite spec bound (class_pos={})",
                c.class_pos
            );
            continue;
        }
        let k = fail[c.class_pos];
        if v > dj[k] {
            report.best_gain = report.best_gain.max(v - dj[k]);
            dj[k] = v;
            improved_classes.insert(k);
            winners.insert(c.class_pos, (fingerprint(&c.moves), c.origin));
        }
    }
    Ok(Some((pass, spec)))
}

/// `#spec-alpha`: refine `dj` in place with a per-start-node (per-class) alpha
/// axis. `dj` is only ever RAISED, so every early return leaves a valid
/// published root.
///
/// Infallible by design: a refusal degrades to the dark answer exactly, and a
/// pass error must never cost the caller its verdict.
///
/// # The two candidate families, and why the split is in the telemetry
///
/// The PRIMARY candidate for each selected class is
/// `clamp(alpha_incumbent + eta * sign(g_k), 0, 1)` with `g_k` that class's own
/// supergradient at the PUBLISHED alpha. That is the per-start-node axis: two
/// classes whose incoming coefficients disagree in sign at a shared unstable
/// neuron get opposite steps there.
///
/// The SECONDARY family is the closed-form ladder, which is a GLOBAL UNIFORM
/// alpha — the same move list for every class. Its per-class content is
/// SELECTION ONLY, and that family has already been measured on this lane as
/// `#interm-alpha` (+0.0025 total). Both families go into ONE pass as separate
/// columns; [`SpecAlphaReport::incumbent_grad_improvements`] says which one
/// actually paid, so a null cannot be read as evidence about the wrong one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refine(
    eng: &BackwardEngine<'_>,
    net: &TwinNet,
    t: usize,
    pol: SpecAlphaPolicy,
    gates: &HeadGates,
    ybox: &YBox,
    fail: &[usize],
    fail_classes: &[usize],
    dj: &mut [f64],
) -> SpecAlphaReport {
    let t0 = Instant::now();
    let root = eng.root;
    let unstable_total: usize = root.layers.iter().map(|l| l.unst.len()).sum();
    // FIRST statement, before any refusal can fire: an inert arming is one grep
    // away. (Pattern: `[interm-alpha] armed:`, which prints before any layer can
    // decline.)
    eprintln!(
        "[spec-alpha] armed: width={} cands={} rounds={} entries_cap={} fail={} unstable={} layers={} ladder={} etas={}",
        pol.width,
        pol.cands,
        pol.rounds,
        pol.entries,
        fail.len(),
        unstable_total,
        root.layers.len(),
        SPEC_ALPHA_LADDER.len(),
        SPEC_ALPHA_ETA.len()
    );
    let root_before = fail.iter().map(|&k| dj[k]).fold(f64::INFINITY, f64::min);
    // `fail` was computed BEFORE the caller folded its own direct-pass bounds
    // into `dj`, so some of `fail` is ALREADY closed on entry. Capture that
    // count here and report the DELTA, or this axis is credited with the
    // unmodified direct pass's closures.
    let closed_before = fail
        .iter()
        .filter(|&&k| super::bab::closed(root.mode, dj[k]))
        .count();
    // The published value per class, frozen before any fold. Round 2's base
    // selection compares candidates against THIS, not against the running
    // `dj[k]` — otherwise a round-1 winner (whose value IS `dj[k]` after the
    // fold) could never displace the incumbent under a strict `>`.
    let dj_entry: Vec<f64> = dj.to_vec();
    let mut report = SpecAlphaReport {
        root_before,
        root_after: root_before,
        ..Default::default()
    };

    if fail.is_empty() {
        eprintln!("[spec-alpha] refuse: no failing classes");
        return report;
    }
    if unstable_total == 0 {
        eprintln!("[spec-alpha] refuse: no unstable neurons");
        return report;
    }
    let cap = pol.entries.min(SPEC_ALPHA_MAX_ENTRIES);
    if cap == 0 {
        eprintln!("[spec-alpha] refuse: entries_cap=0");
        return report;
    }
    if pol.deadline.is_some_and(|dl| Instant::now() >= dl) {
        eprintln!("[spec-alpha] refuse: deadline");
        return report;
    }

    // Hardest failing classes first; class id ascending is the data-independent
    // tie-break, so selection is deterministic.
    let mut order: Vec<usize> = (0..fail.len()).collect();
    order.sort_by(|&a, &b| {
        dj[fail[a]]
            .partial_cmp(&dj[fail[b]])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(fail_classes[a].cmp(&fail_classes[b]))
    });
    order.truncate(pol.width.max(1));
    report.classes_selected = order.len();

    let e_grad = pol.cands.min(SPEC_ALPHA_ETA.len()).max(1);
    let e_ladder = pol.cands.min(SPEC_ALPHA_LADDER.len()).max(1);

    // ---- PRIMARY: the per-class step from the INCUMBENT. One extra pass.
    let incumbent = match incumbent_grads(eng, net, t, gates, ybox, fail_classes, &order) {
        Ok(inc) => {
            report.passes += 1;
            report.incumbent_grads_ok = inc.grads.len();
            eprintln!(
                "[spec-alpha] incumbent: cols={} selected={} grads_ok={} grads_none={} unst_rows={} secs={:.2}",
                fail_classes.len(),
                order.len(),
                inc.grads.len(),
                order.len() - inc.grads.len(),
                inc.pass.unst_rows.as_ref().map_or(0, std::collections::BTreeMap::len),
                t0.elapsed().as_secs_f64()
            );
            Some(inc)
        }
        Err(error) => {
            // Degrades to the ladder-only mechanism, LOUDLY: a run with this
            // line is not evidence about the per-start-node axis.
            eprintln!("[spec-alpha] refuse: incumbent pass: {error}");
            None
        }
    };

    // ---- SECONDARY: the global-uniform ladder. A pure function of the root
    // gates, so build it ONCE and replicate per selected class.
    let ladder: Vec<(usize, Vec<(usize, usize, f64)>)> = (0..e_ladder)
        .map(|e| (e, ladder_moves(root, e)))
        .filter(|(_, m)| !m.is_empty())
        .collect();

    let mut raw: Vec<(usize, CandOrigin, Vec<(usize, usize, f64)>)> = Vec::new();
    // Gradient candidates FIRST: `assign_columns` drops whole candidates from
    // the END, so under a tight cap the axis survives and the already-measured
    // ladder is what gets dropped.
    if let Some(inc) = incumbent.as_ref() {
        for &class_pos in &order {
            let Some(grad) = inc.grads.get(&class_pos) else {
                continue;
            };
            for (ei, &eta) in SPEC_ALPHA_ETA.iter().take(e_grad).enumerate() {
                let moves = gradient_moves(root, &inc.alpha, grad, eta);
                if !moves.is_empty() {
                    raw.push((class_pos, CandOrigin::IncumbentGrad(ei), moves));
                }
            }
        }
    }
    for &class_pos in &order {
        for (e, moves) in &ladder {
            raw.push((class_pos, CandOrigin::Ladder(*e), moves.clone()));
        }
    }
    if raw.is_empty() {
        eprintln!("[spec-alpha] refuse: no candidate produced a move");
        return report;
    }
    let (cands, entries, truncated) = assign_columns(raw, cap);
    if cands.is_empty() {
        eprintln!("[spec-alpha] refuse: entries_cap truncated all candidates");
        report.truncated += truncated;
        return report;
    }

    let mut improved_classes: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut winners: Winners = Winners::new();
    let want_grad = pol.rounds >= 2;
    // Bind the outcome BEFORE the match so the shared borrow of `cands` ends
    // here and `cands` can be moved into `prev_cands` below.
    let outcome = run_round(
        eng,
        net,
        t,
        1,
        &cands,
        entries,
        truncated,
        fail,
        fail_classes,
        gates,
        ybox,
        want_grad,
        dj,
        &mut improved_classes,
        &mut winners,
        &mut report,
    );
    let round1 = match outcome {
        Ok(Some(v)) => Some(v),
        Ok(None) => None,
        Err(error) => {
            eprintln!("[spec-alpha] refuse: pass: {error}");
            None
        }
    };

    // ---- Gradient rounds. The axis keeps ascending rather than enumerating.
    if let Some((pass, spec)) = round1 {
        let mut prev_pass = pass;
        let mut prev_spec = spec;
        let mut prev_cands = cands;
        for round in 2..=pol.rounds.max(1) {
            if pol.deadline.is_some_and(|dl| Instant::now() >= dl) {
                eprintln!("[spec-alpha] refuse: deadline");
                break;
            }
            // Per class: the best base so far, stepped by THAT class's OWN
            // supergradient.
            //
            // SEEDED WITH THE INCUMBENT, at the class's PUBLISHED value. Without
            // that seed a round in which no candidate beat the incumbent still
            // hill-climbs from the best LOSER — a deliberately perturbed
            // global-uniform alpha that is strictly worse than the point the
            // root actually published.
            let mut best_by_class: std::collections::BTreeMap<usize, (Option<usize>, f64)> =
                std::collections::BTreeMap::new();
            if incumbent.is_some() {
                for &class_pos in &order {
                    best_by_class.insert(class_pos, (None, dj_entry[fail[class_pos]]));
                }
            }
            for (idx, c) in prev_cands.iter().enumerate() {
                let v = prev_spec[c.col];
                if !v.is_finite() {
                    continue;
                }
                match best_by_class.entry(c.class_pos) {
                    std::collections::btree_map::Entry::Vacant(slot) => {
                        slot.insert((Some(idx), v));
                    }
                    std::collections::btree_map::Entry::Occupied(mut slot) => {
                        if v > slot.get().1 {
                            slot.insert((Some(idx), v));
                        }
                    }
                }
            }
            let mut raw: Vec<(usize, CandOrigin, Vec<(usize, usize, f64)>)> = Vec::new();
            let mut from_incumbent = 0usize;
            for (&class_pos, &(idx, _)) in &best_by_class {
                // `None` means the incumbent is still this class's best base:
                // step from the published alpha, using the incumbent pass's own
                // column for this class.
                let (base, base_pass, base_col, tag) = match (idx, incumbent.as_ref()) {
                    (Some(i), _) => {
                        let c = &prev_cands[i];
                        (c.full_alpha(root), &prev_pass, c.col, false)
                    }
                    (None, Some(inc)) => {
                        from_incumbent += 1;
                        (inc.alpha.clone(), &inc.pass, class_pos, true)
                    }
                    (None, None) => continue,
                };
                let Some(grad) =
                    super::alpha_opt::supergradient(net, root, &base, base_pass, base_col)
                else {
                    continue;
                };
                for (ei, &eta) in SPEC_ALPHA_ETA.iter().take(e_grad).enumerate() {
                    let moves = gradient_moves(root, &base, &grad, eta);
                    if !moves.is_empty() {
                        let origin = if tag {
                            CandOrigin::IncumbentGrad(ei)
                        } else {
                            CandOrigin::RoundGrad(ei)
                        };
                        raw.push((class_pos, origin, moves));
                    }
                }
            }
            if raw.is_empty() {
                eprintln!("[spec-alpha] refuse: gradient round {round} produced no moves");
                break;
            }
            let (cands_n, entries_n, truncated_n) = assign_columns(raw, cap);
            if cands_n.is_empty() {
                eprintln!("[spec-alpha] refuse: entries_cap truncated all candidates");
                report.truncated += truncated_n;
                break;
            }
            eprintln!(
                "[spec-alpha] round {round}: bases={} from_incumbent={from_incumbent} cands={}",
                best_by_class.len(),
                cands_n.len()
            );
            let want_grad = round < pol.rounds;
            let outcome = run_round(
                eng,
                net,
                t,
                round,
                &cands_n,
                entries_n,
                truncated_n,
                fail,
                fail_classes,
                gates,
                ybox,
                want_grad,
                dj,
                &mut improved_classes,
                &mut winners,
                &mut report,
            );
            match outcome {
                Ok(Some((p, s))) => {
                    prev_pass = p;
                    prev_spec = s;
                    prev_cands = cands_n;
                }
                Ok(None) => break,
                Err(error) => {
                    eprintln!("[spec-alpha] refuse: pass: {error}");
                    break;
                }
            }
            if !want_grad {
                break;
            }
        }
    }

    report.classes_improved = improved_classes.len();
    report.distinct_winners = winners
        .values()
        .map(|&(fp, _)| fp)
        .collect::<std::collections::BTreeSet<u64>>()
        .len();
    // Count every NON-LADDER winner, not just `IncumbentGrad`. At `rounds >= 2`
    // a class whose winner came from the axis and was then improved again in
    // round 2 carries `RoundGrad`, so the old filter reported the field's own
    // opposite in exactly the arm it is read in: the axis working better looked
    // like the axis not working. The ladder is the only rival family, so
    // "not Ladder" is the honest attribution.
    report.incumbent_grad_improvements = winners
        .values()
        .filter(|&&(_, origin)| !matches!(origin, CandOrigin::Ladder(_)))
        .count();
    report.root_after = fail.iter().map(|&k| dj[k]).fold(f64::INFINITY, f64::min);
    let closed_after = fail
        .iter()
        .filter(|&&k| super::bab::closed(root.mode, dj[k]))
        .count();
    report.closed_extra = closed_after.saturating_sub(closed_before);
    eprintln!(
        "[spec-alpha] done: rounds_run={} passes={} entries={} classes_selected={} \
classes_improved={} incumbent_grads_ok={} distinct_winners={} incumbent_grad_improvements={} \
root_before={:.6} root_after={:.6} delta={:+.6} best_gain={:.6} \
closed_before={closed_before} closed_after={closed_after} closed_extra={} truncated={} secs={:.2}",
        report.rounds_run,
        report.passes,
        report.entries,
        report.classes_selected,
        report.classes_improved,
        report.incumbent_grads_ok,
        report.distinct_winners,
        report.incumbent_grad_improvements,
        report.root_before,
        report.root_after,
        report.root_after - report.root_before,
        report.best_gain,
        report.closed_extra,
        report.truncated,
        t0.elapsed().as_secs_f64()
    );
    report
}
