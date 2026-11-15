// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #alpha-opt: projected supergradient optimization of the root trunk alpha
//! (#twinwall).
//!
//! WHY. The lane's lower-line slopes are born binary — `gates_from_box`
//! (`root.rs`) pins `alpha = [u >= -l]` per unstable neuron and nothing ever
//! moves them. alpha-beta-CROWN OPTIMIZES the same free parameter, and on the
//! cifar100 deep band (lane roots -0.61..-1.76) it proves at the root rows this
//! lane starts too deep to close (idx_7704: lane root -0.35, abc INIT=1 with
//! zero BaB). Every row the lane has ever proven started shallower than -0.57,
//! so the root bound is the binding constraint — this module moves it.
//!
//! WHAT. A root-phase loop (before the tree starts, never per-domain):
//! run one certified margin-seeded Lower pass under a candidate alpha, extract
//! a per-neuron supergradient from the SAME pass, step alpha toward a better
//! bound (projected to `[0, 1]`, restricted to unstable neurons), and keep the
//! best CERTIFIED objective ever seen (monotone accept — the committed alpha is
//! never worse than the binary heuristic under the frozen evaluation frame).
//!
//! THE GRADIENT (exact for the frozen-sign linearization; heuristic overall).
//! At trunk relu `li` the Lower-lane transform is `lv = vp*alpha_j + vn*s_j`
//! with `vp = max(v, 0)`, `vn = min(v, 0)` (`engine.rs`), so alpha enters ONLY
//! through the positive part: `d lv / d alpha_j = vp >= 0`. Holding fixed (a)
//! the sign selections the pass made at every other relu and (b) the concretize
//! argmin signs, the final bound is LINEAR in `lv`, and its coefficient is the
//! value at the argmin point `x*` of neuron `j`'s pre-activation under the
//! pass's own selected lines:
//!
//!   d bound_f / d alpha_{li,j} = vp_{li,j,f} * U_{li,j,f},
//!
//! where `x*_i = mid_i - sign(A_{i,f}) * rad_i` (the derivative of the
//! concretize `A@mid - |A|@rad` term is `mid_i - sign(A_i)*rad_i`, i.e. it
//! FLIPS with the coefficient sign — that is exactly what seeding the walk at
//! `x*` accounts for), and `U` is one cheap single-column forward walk of `x*`
//! through the linearized trunk (each earlier relu replaced by the affine line
//! the pass selected for that column: lower `alpha*z` on `v > 0`, upper chord
//! `s*z + c` on `v < 0`). Across sign flips the objective is only piecewise —
//! we do NOT chase the exact Clarke supergradient (alpha-beta-CROWN does not
//! either) — and the certified-slack terms' own alpha-dependence (the E-lane
//! grows with `ms = max(alpha, s)` when alpha exceeds `s`, plus the gamma
//! envelopes) is IGNORED in the direction: the direction is the gradient of
//! the frozen-sign LINEARIZATION, not of the slack-included certified bound.
//! Both approximations are direction-only: a wrong direction wastes one pass
//! and nothing more, because every ACCEPTED iterate is scored by the
//! unchanged certified Outward pass, slack included.
//!
//! SOUNDNESS.
//! * `alpha*y <= relu(y)` holds pointwise on all of R for ANY `alpha in
//!   [0, 1]` — validity of the lower line is box-independent and does not
//!   require binariness (the GPU seam's own docs state the same generality).
//! * The upper chord `(s, c)` depends only on `(l, u)`, which this module
//!   never moves — it is NOT touched (the build's `repair_upper_lines`
//!   certificate stands verbatim).
//! * THE ONE LOAD-BEARING COUPLING: the backward error lane contracts by
//!   `ms = max(alpha, s)` — the Lipschitz constant of `v -> vp*a + vn*s`.
//!   A stale binary-era `ms` under a raised alpha would UNDER-scale the carried
//!   error (a false-UNSAT generator), so `ms` is re-derived from the candidate
//!   alpha on every evaluation and on commit, by the same `max(alpha, s)`
//!   formula the root uses.
//! * The forward tableau is NOT rebuilt: the stored `l`/`u`/`clip_rows` were
//!   produced by a valid relaxation and remain valid enclosures of the true
//!   pre-activation ranges regardless of which (valid) backward alpha is used
//!   later. In particular `apply_gates`' outward widening — whose HISTORICAL
//!   constant's derivation assumes a binary lower product — never sees a
//!   fractional alpha through this module, because backward-only optimization
//!   never executes it. (#rebake DOES bake fractional alpha forward; for that
//!   path `apply_gates` carries its own magnitude-scaled widening branch with
//!   the re-derivation in its header doc.)
//! * Search iterates evaluate through `DomainGates` overrides (the engine's
//!   established injection point); the root struct is mutated ONCE, on accept.
//!   With the gate off, on refusal before a certified candidate, or on
//!   no-improvement, the root is byte-identical to the build. A later pass
//!   error stops new candidates but may still publish the best earlier,
//!   strictly improving certified candidate.
//!
//! GATE. The scored [`super::RootAlphaRuntimePolicy`] is frozen once from the
//! typed preset/LeverSet; its exact legacy `NY_MARGIN_ROW_ALPHA_OPT=1` overlay
//! also arms it, and the declaration default is OFF. The frozen iteration
//! control (default 20) caps iterations; the frozen wall request (default 20,
//! further capped to 40% of the remaining deadline) caps wall clock. Engagement telemetry
//! (`[alpha-opt] iter=...`) prints unconditionally when armed, so an inert
//! arming is detectable in one log (a null from a lever that never fired is
//! vacuous).
//!
//! BANKED CAVEAT (why this is pass-frugal): on the INTERNAL verifier a working
//! alpha gradient measured 0 conversions because cost-per-iteration was
//! binding. Here one iteration costs ONE backward pass (~600 ms on cifar100)
//! plus a single-column forward walk (milliseconds); the y-pack and head frame
//! are built once and frozen.

use ndarray::Array2;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::Instant;

use super::bounds::{
    compose_viay, head_gates_with, margin_seed, per_class_direct, row_dots, HeadGates, MarginBatch,
    MarginSeed, YBox,
};
use super::engine::{BackwardEngine, Collect, DomainGates, GateVecs, LaneDir, PassOut};
use super::joint_grad::{box_weights, joint_adjoint, JointFrame, JOINT_GRAD, JOINT_GRAD_PROBE};
use super::net::{conv_apply_forward, TwinNet, TwinOp};
use super::root::RootGates;

/// Fully resolved root-alpha optimizer controls. `Default` deliberately means
/// "resolve the compatibility layered process" for direct tests and research
/// callers. Scored execution always uses [`AlphaOptCfg::from_policy`] so the
/// optimizer cannot reopen a process-global authority after the flight receipt
/// was frozen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AlphaOptCfg {
    pub(crate) iters: usize,
    pub(crate) secs: f64,
    /// Adam step. Zero selects the historical normalized-supergradient path.
    pub(crate) lr: f64,
    /// Number of worst frozen-frame columns averaged for steering.
    pub(crate) cols: usize,
    /// Revert unless the independently recomputed published root improves.
    pub(crate) guard: bool,
    /// #router-r3 `rule.governor`: an explicit yield ratio (tests, research
    /// harnesses). `None` — every production constructor — resolves the
    /// process's stop authority (armed only under `optimizer.mode: on`), else
    /// the shipped `marginal::GOVERNOR_RATIO` (`None`: today's ascent,
    /// byte-identical).
    pub(crate) governor: Option<f64>,
    /// #router-r3 `rule.ia_yield`: interm-phase time the ascent's CONTRACT
    /// excludes from the remaining window before its share is taken, so time
    /// the interm candidates freed reaches the tree instead of inflating the
    /// ascent. `ZERO` — every production constructor except the standalone
    /// entry under an armed `rule.ia_yield` — is today's budget, exactly.
    pub(crate) contract_credit: std::time::Duration,
}

impl Default for AlphaOptCfg {
    fn default() -> Self {
        Self::from_policy(super::RootAlphaRuntimePolicy::from_layered_process())
    }
}

impl AlphaOptCfg {
    pub(super) const fn from_policy(policy: super::RootAlphaRuntimePolicy) -> Self {
        Self {
            iters: policy.alpha_iters,
            secs: policy.alpha_secs,
            lr: policy.alpha_lr,
            cols: policy.alpha_cols,
            guard: policy.alpha_guard,
            governor: None,
            contract_credit: std::time::Duration::ZERO,
        }
    }

    /// Controls shared by the joint re-bake ascent. Re-bake owns its round
    /// iteration/wall policy and performs the published-frame guard after the
    /// rebuild, so only the two steering controls are resolved here. Keeping
    /// this constructor explicit prevents discarded reads of the standalone
    /// optimizer's iters/secs/guard axes from polluting attribution receipts.
    pub(super) const fn for_rebake(
        policy: super::RootAlphaRuntimePolicy,
        iters: usize,
        secs: f64,
    ) -> Self {
        Self {
            iters,
            secs,
            lr: policy.alpha_lr,
            cols: policy.alpha_cols,
            guard: false,
            governor: None,
            contract_credit: std::time::Duration::ZERO,
        }
    }
}

/// Cheap, side-effect-free admission fact shared by the standalone and rebake
/// execution receipts.  It deliberately excludes budget/data-path failures:
/// those require a completed report before T1 can award armed engagement.
/// #alpha-frac: fraction of the REMAINING deadline the root alpha ascent may
/// spend. Shipped at 0.4, byte-identical to the historical hardcoded literal.
///
/// Raising it trades tree time for root-bound quality. The 2026-09-16 evidence
/// says that trade is favourable on cifar100 resnet_large — the tree is not
/// where either verifier's budget decides the row — but it is UNMEASURED, so the
/// default does not change.
pub(crate) const ALPHA_REMAINING_FRAC: f64 = 0.4;

/// `#budget-governor`: the root alpha ascent's yield gate.
///
/// SUPERSEDES an earlier absolute-epsilon form (`ASCENT_MARGINAL_STOP`). An
/// absolute threshold is a per-benchmark constant in disguise — it depends on
/// the scale of the objective, so it would need re-tuning per benchmark and per
/// row family. `MarginalYield` uses RELATIVE self-decay instead: yield when this
/// phase's marginal gain falls below `ratio` of its OWN best. Unit-free, and the
/// SAME ratio governs `backward_interm`, whose rates are four orders of
/// magnitude larger. See `marginal.rs` for the measured verification of both.
use super::marginal::{GovernorSource, MarginalYield};

/// #channel-alpha: share one alpha per CONV CHANNEL instead of per neuron.
///
/// # Why
///
/// MEASURED 2026-09-16 on cifar100 resnet_large: the ascent commits **2797
/// per-neuron alphas and pins 2673 of them (95.57%) to the clamp**, converges by
/// iteration 18 of 20 (last three steps rejected), and is completely insensitive
/// to budget — `alpha_secs` 20->34 and the remaining-fraction 0.4->0.7 both left
/// the root bound BIT-IDENTICAL on 4/4 rows. It is SATURATED, not starved.
///
/// alpha-beta-CROWN shares alpha per channel on this benchmark
/// (`full_conv_alpha: False` -> `channel_only=True`,
/// `auto_LiRPA/auto_LiRPA/backward_bound.py:899`) and AUTO-DEGRADES into
/// channel-only above 512 unstable neurons (`:870`, `:901-905`) — precisely ny's
/// regime. Its ascent buys +0.1218 of root bound to ny's +0.063 on the same axis,
/// from an identical root lower line.
///
/// Collapsing C*H*W free variables to C gives each far more gradient signal and
/// removes the pinning. ny HAS this in the graph lane
/// (`network/graph_alpha/bounds/alpha.rs:804`) and NOT in margin-row, which is
/// the lane that proves cifar100 — the 14th wrong-lane instance.
///
/// # Soundness
///
/// `alpha in [0, 1]` per neuron is what makes each line a valid DeepPoly lower
/// relaxation. A channel-shared assignment is a RESTRICTION to a subspace of
/// already-valid per-neuron alphas — every value it can produce was reachable
/// before. It cannot create an invalid line, and the published bound is still
/// scored by the same certified pass and kept only when it improves.
///
/// `false` ships the historical per-neuron ascent, byte-identical.
pub(crate) const CHANNEL_ALPHA: bool = false;

/// #steer-branch. The ascent's objective for a class is `d = max(direct, m1,
/// m2v)`, where `m1`/`m2v` come from the FROZEN evaluation frame and only
/// `direct` moves with alpha. `d` is therefore a MAX, and the supergradient of
/// a max is the gradient of the branch that ATTAINS it. `supergradient`
/// differentiates `direct` alone, so on a class where a frozen term attains the
/// max, `d` is locally CONSTANT in alpha and its true supergradient is ZERO —
/// yet the historical code still folds that class's `d(direct)/d(alpha)` into
/// the mean direction. Because the steer set is the LOWEST-`d` classes, it is
/// biased toward exactly the classes most likely to be frozen-pinned.
///
/// Armed, a frozen-pinned class contributes nothing to the mean (its true
/// supergradient), and an ascent whose every open class is pinned stops instead
/// of proposing iterates that cannot move the objective.
///
/// This changes the DIRECTION only. `worst` — the published objective — is
/// `scored[0].0` either way, every iterate is still re-scored by the unchanged
/// certified `eval`, and under `cfg.guard` the whole ascent is still reverted
/// unless the independently recomputed published root improves. Soundness is
/// untouched by construction.
pub(crate) const STEER_ACTIVE_BRANCH: bool = false;

/// #head-alpha: ascend the HEAD relu's lower slopes jointly with the trunk's.
///
/// # Why
///
/// MEASURED 2026-09-23 with a scratch copy of alpha-beta-CROWN on cifar100
/// `idx_7258` (initial CROWN -0.829), all arms with intermediates frozen at the
/// initial CROWN bounds, i.e. ny's own frame: abc optimizing every alpha
/// reaches -0.6723; the same run with ONLY the head relu's alpha held at its
/// initial value reaches -0.7267; ny's frozen-frame ascent reaches -0.741. So
/// 0.054 of abc's 0.157 frozen-frame gain is the head slope, and ny — which
/// holds the head at the binary area heuristic (`bounds::head_gates`) and never
/// moves it — is within 0.014 of abc's head-frozen arm. abc's
/// `share_alphas` arm (one alpha per start node across all spec rows, head
/// included) lost only 0.03 of 0.27 against per-row alphas, so ONE shared head
/// slope per neuron captures nearly all of it; that is what this ascends.
///
/// # Mechanism
///
/// The direct objective of class `f` is `min_x S_f . L(x)` with seed
/// `S[j, f] = wp[f, j] * alpha_j + wn[f, j] * s_j`. At the concretization argmin
/// `x*` and with the pass's relaxation choices frozen, `L(x*)` is exactly the
/// frozen-sign linearized walk's head pre-activation `y(x*)`
/// (`linearized_walk(.., head = true)`, the walk #margin-row-beta C3 already
/// uses), so `d(direct_f) / d(alpha_j) = wp[f, j] * y_j(x*)` — a supergradient
/// of a concave piecewise-linear function, same contract as the trunk's.
///
/// # Soundness
///
/// Direction only. Every candidate is scored by the unchanged certified pass
/// with the candidate head gates, the published bound is recomputed from
/// scratch by `root_eval` (which reads the committed `RootGates::head_alpha`
/// through `head_gates_with`), and under `cfg.guard` the ascent is reverted
/// unless that published bound improves. The slopes are clamped to `[0, 1]`
/// and consumed only on head neurons unstable in the evaluated box, where any
/// such slope is a valid lower line.
pub(crate) const HEAD_ALPHA: bool = true;

/// Pick the steer set from the objective-sorted class list, and count how many
/// classes are frozen-pinned over the WHOLE set (a census, not a window count).
///
/// `scored` is `(d, class, direct_attains_the_max)` sorted by `d` ascending.
/// Unarmed this is exactly the historical prefix, so the disarmed path is
/// arithmetically identical.
fn steer_selection(scored: &[(f64, usize, bool)], want: usize, armed: bool) -> (Vec<usize>, usize) {
    let pinned = scored.iter().filter(|s| !s.2).count();
    let cols: Vec<usize> = if armed {
        scored
            .iter()
            .filter(|s| s.2)
            .take(want)
            .map(|s| s.1)
            .collect()
    } else {
        scored.iter().take(want).map(|s| s.1).collect()
    };
    (cols, pinned)
}

/// Spatial group size (`H*W`) per trunk-relu layer, or `None` where the layer is
/// not conv-shaped (a Gemm relu has no channel structure to share over).
///
/// Derived from the producing [`ConvOp`]'s `oshape` `(C, H, W)`: neuron `j`
/// belongs to channel `j / (H*W)`.
fn channel_group_sizes(net: &TwinNet) -> Vec<Option<usize>> {
    net.trunk_relus
        .iter()
        .map(|&ri| {
            let TwinOp::Relu { input, .. } = net.ops.get(ri)? else {
                return None;
            };
            // Op `k` produces tensor `k + 1` (see `net.tsize[k + 1]`), so the op
            // producing tensor `input` is `ops[input - 1]`.
            let TwinOp::Conv(c) = net.ops.get(input.checked_sub(1)?)? else {
                return None;
            };
            let (_c, h, w) = c.oshape;
            // A 1x1 spatial map has no channel structure to share over.
            (h * w > 1).then_some(h * w)
        })
        .collect()
}

/// Project a per-neuron step onto the channel-constant subspace: every neuron in
/// a channel receives that channel's MEAN step. Returns the number of neurons
/// whose step was altered (0 when the layer has no channel structure).
fn project_step_to_channels(steps: &mut [f64], live: &[usize], group: usize) -> usize {
    if group <= 1 {
        return 0;
    }
    let mut sum: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
    for &j in live {
        let e = sum.entry(j / group).or_insert((0.0, 0));
        e.0 += steps[j];
        e.1 += 1;
    }
    let mut changed = 0usize;
    for &j in live {
        if let Some(&(tot, n)) = sum.get(&(j / group)) {
            if n > 0 {
                let mean = tot / n as f64;
                if steps[j] != mean {
                    steps[j] = mean;
                    changed += 1;
                }
            }
        }
    }
    changed
}

pub(crate) fn execution_eligible(root: &RootGates, adv: &[usize]) -> bool {
    root.mode.outward() && !adv.is_empty() && root.layers.iter().any(|layer| !layer.unst.is_empty())
}

/// What one optimization run did (telemetry + test surface).
pub struct AlphaOptReport {
    pub(crate) cfg: AlphaOptCfg,
    /// Effective wall slice after the 40%-of-remaining cap.
    pub effective_secs: f64,
    /// Whether the instance deadline, rather than the configured request,
    /// reduced the effective wall slice.
    pub deadline_capped: bool,
    /// Whether the next certified pass was refused by the effective wall cap.
    pub budget_exhausted: bool,
    /// Whether all configured iterations completed their candidate pass.
    pub iters_cap_exhausted: bool,
    /// Largest number of frozen-frame columns actually used by a gradient.
    pub max_steer_cols: usize,
    /// Published-frame baseline evaluations owned by the guard.
    pub guard_baseline_evals: usize,
    /// Published-frame candidate evaluations owned by the guard.
    pub guard_candidate_evals: usize,
    /// Guard-triggered rollbacks.
    pub guard_reversions: usize,
    /// Frozen-frame failing objective columns available to steering.
    pub fail_classes: usize,
    /// Candidate updates applied before a certified evaluation.
    pub candidate_steps: usize,
    /// Candidate updates that used the legacy normalized step.
    pub legacy_steps: usize,
    /// Candidate updates that used Adam.
    pub adam_steps: usize,
    /// Candidate evaluations that strictly improved the frozen objective.
    pub frozen_improvements: usize,
    /// Candidate commits for which the guard had a publish/revert decision.
    pub guard_opportunities: usize,
    /// #alpha-saturation PC-2: unstable neurons whose COMMITTED alpha sits at
    /// exactly 0.0 or 1.0 — i.e. on the `[0,1]` clamp boundary.
    ///
    /// THE DIAGNOSIS THIS SETTLES. The campaign has been asserting that "the
    /// ascent saturates by iteration 11-12 and the binding constraint is the
    /// clamp", and using that to argue no richer alpha axis can help. That was
    /// inference, never instrumented. If the committed alphas are overwhelmingly
    /// at the vertices, the ascent is degenerating to exactly the vertex set
    /// `start_node_gates` policy 1 already enumerates, and a per-start-node
    /// ascent would buy nothing. If they are interior, the clamp is NOT binding
    /// and the premise is false.
    pub saturated_alphas: usize,
    /// Unstable neurons whose committed alpha was examined (the denominator).
    pub committed_alphas: usize,
    /// Published baseline already computed by the guard, when present.
    pub published_before: Option<f64>,
    /// Published candidate already computed by the guard, when finite.
    pub published_after: Option<f64>,
    /// Certified frozen-frame objective of the heuristic (binary) alpha.
    pub baseline: f64,
    /// Best certified frozen-frame objective found (>= baseline by monotone
    /// accept; == baseline when nothing improved).
    pub best: f64,
    /// Gradient iterations attempted (excludes the baseline pass).
    pub iters: usize,
    /// Certified backward passes spent (baseline + iterations).
    pub passes: usize,
    /// Was a non-heuristic alpha committed to the root?
    pub accepted: bool,
    /// Neurons whose committed alpha differs from the heuristic.
    pub moved_neurons: usize,
    /// Largest committed |delta alpha|.
    pub max_dalpha: f64,
    /// #router-r3 `rule.governor`: the iteration at which the ascent's own
    /// relative self-decay rule yielded (`None`: it did not).
    pub governor_yield_iter: Option<usize>,
    /// Seconds of the ascent's budget the yield handed downstream.
    pub governor_freed_secs: f64,
    /// #router-r3 `rule.ia_yield`: interm seconds the ascent's contract
    /// excluded from its window (zero: today's budget).
    pub contract_credit_secs: f64,
}

/// Closed root-phase execution evidence for the jointly governed standalone
/// alpha and rebake controls.  Configuration is copied from the immutable
/// receipt-owned policy; observations are populated only from reports returned
/// by the actual dispatches.  Tree code can carry this value but never reads it
/// to make a bound or scheduling decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootAlphaExecutionEvidence {
    pub alpha_enabled: bool,
    pub alpha_iters: usize,
    pub alpha_secs: f64,
    pub alpha_lr: f64,
    pub alpha_cols: usize,
    pub alpha_guard: bool,
    pub rebake_enabled: bool,
    pub rebake_rounds: usize,
    pub rebake_secs: f64,
    pub eligible: bool,
    pub standalone_suppressed: usize,
    pub standalone_attempted: usize,
    pub standalone_completed: usize,
    pub standalone_effective_secs: f64,
    pub standalone_deadline_capped: bool,
    pub standalone_budget_exhausted: bool,
    pub standalone_iters_cap_exhausted: bool,
    pub standalone_fail_classes: usize,
    pub standalone_iters_attempted: usize,
    pub standalone_candidate_steps: usize,
    pub standalone_legacy_steps: usize,
    pub standalone_adam_steps: usize,
    pub standalone_passes: usize,
    pub standalone_max_steer_cols: usize,
    pub standalone_frozen_improvements: usize,
    pub standalone_accepted: bool,
    pub standalone_moved_neurons: usize,
    pub standalone_guard_opportunities: usize,
    pub standalone_guard_baseline_evals: usize,
    pub standalone_guard_candidate_evals: usize,
    pub standalone_guard_reversions: usize,
    pub standalone_frozen_before_bits: Option<u64>,
    pub standalone_frozen_best_bits: Option<u64>,
    pub standalone_published_before_bits: Option<u64>,
    pub standalone_published_after_bits: Option<u64>,
    pub rebake_suppressed: usize,
    pub rebake_attempted: usize,
    pub rebake_completed: usize,
    pub rebake_effective_secs: f64,
    pub rebake_deadline_capped: bool,
    pub rebake_budget_exhausted: bool,
    pub rebake_round_cap_exhausted: bool,
    pub rebake_entry_unbaked_alpha: bool,
    pub rebake_baseline_evals: usize,
    pub rebake_rounds_started: usize,
    pub rebake_alpha_completed: usize,
    pub rebake_alpha_iters_attempted: usize,
    pub rebake_alpha_candidate_steps: usize,
    pub rebake_alpha_legacy_steps: usize,
    pub rebake_alpha_adam_steps: usize,
    pub rebake_alpha_passes: usize,
    pub rebake_alpha_max_fail_classes: usize,
    pub rebake_alpha_max_steer_cols: usize,
    pub rebake_alpha_moved_neurons: usize,
    pub rebake_rebuilds_completed: usize,
    pub rebake_rounds_scored: usize,
    pub rebake_rounds_accepted: usize,
    pub rebake_fixed_point_stops: usize,
    pub rebake_affordability_refusals: usize,
    pub rebake_aborts: usize,
    pub rebake_published_before_bits: Option<u64>,
    pub rebake_published_after_bits: Option<u64>,
    /// #spec-alpha policy echo. Built in the root driver immediately after
    /// `#rebake` returns, i.e. BEFORE `#spec-alpha` runs inside
    /// `bab::root_eval` — so at construction this receipt can carry
    /// configuration only. The YIELD is folded in later by
    /// [`Self::with_spec_alpha_yield`], at the site where
    /// [`super::bab::RootEval`] and the outgoing stats are both in scope.
    pub spec_alpha_enabled: bool,
    /// See [`Self::spec_alpha_enabled`].
    pub spec_alpha_width: usize,
    /// See [`Self::spec_alpha_enabled`].
    pub spec_alpha_cands: usize,
    /// See [`Self::spec_alpha_enabled`].
    pub spec_alpha_rounds: usize,
    /// See [`Self::spec_alpha_enabled`].
    pub spec_alpha_entries: usize,
    /// #spec-alpha YIELD, populated from [`super::bab::RootEval::spec_alpha`]
    /// at the site that builds the tree's stats — the first point at which the
    /// refinement has run and the receipt is still being assembled.
    ///
    /// `None` on the bit fields means THE AXIS DID NOT RUN (dark, or no failing
    /// class survived the direct pass), which is a different fact from a zero
    /// delta and must not be read as one. Without these the yield is
    /// stderr-only, and `ny benchmarks run` nulls stdout and forces
    /// `RUST_LOG=error` — which is exactly the `#interm-alpha` gap
    /// (`BackwardIntermExecutionEvidence` carries no `interm_alpha` field, so
    /// every sweep through it is uninterpretable by construction).
    pub spec_alpha_ran: bool,
    /// Rounds that issued a certified pass.
    pub spec_alpha_rounds_run: usize,
    /// Certified passes issued, INCLUDING the incumbent collect pass.
    pub spec_alpha_passes: usize,
    /// Exception entries committed across all rounds.
    pub spec_alpha_entries_committed: usize,
    /// Classes given candidates.
    pub spec_alpha_classes_selected: usize,
    /// Selected classes whose INCUMBENT supergradient resolved. `0` here with
    /// `spec_alpha_classes_selected > 0` means the axis degraded to the
    /// global-uniform ladder: VOID for the axis, not a negative about it.
    pub spec_alpha_incumbent_grads_ok: usize,
    /// DISTINCT classes whose `dj` strictly improved, across all rounds.
    pub spec_alpha_classes_improved: usize,
    /// Distinct winning alphas among improving classes. `1` means a rank-1
    /// global candidate would have done as well — read this before `delta`.
    pub spec_alpha_distinct_winners: usize,
    /// Improving classes whose winner came from the per-class gradient family
    /// rather than the global-uniform ladder.
    pub spec_alpha_incumbent_grad_improvements: usize,
    /// Classes moved from failing to closed, as a DELTA against the count
    /// already closed when the refinement was entered.
    pub spec_alpha_closed_extra: usize,
    /// Whole candidates dropped by the entry cap.
    pub spec_alpha_truncated: usize,
    /// `min dj` over the failing set before the refinement.
    pub spec_alpha_root_before_bits: Option<u64>,
    /// `min dj` over the failing set after the refinement.
    pub spec_alpha_root_after_bits: Option<u64>,
    /// Best single-class gain.
    pub spec_alpha_best_gain_bits: Option<u64>,
    pub policy_mismatch: bool,
}

impl RootAlphaExecutionEvidence {
    /// Fold a completed `#spec-alpha` refinement's yield into this receipt.
    ///
    /// Called where [`super::bab::RootEval`] and the outgoing stats are both in
    /// scope. `None` leaves every yield field at its "did not run" value, so a
    /// dark instance and a zero-delta instance are distinguishable in the
    /// receipt without reading stderr.
    #[must_use]
    pub(crate) fn with_spec_alpha_yield(
        mut self,
        report: Option<&super::spec_alpha::SpecAlphaReport>,
    ) -> Self {
        let Some(report) = report else {
            return self;
        };
        self.spec_alpha_ran = true;
        self.spec_alpha_rounds_run = report.rounds_run;
        self.spec_alpha_passes = report.passes;
        self.spec_alpha_entries_committed = report.entries;
        self.spec_alpha_classes_selected = report.classes_selected;
        self.spec_alpha_incumbent_grads_ok = report.incumbent_grads_ok;
        self.spec_alpha_classes_improved = report.classes_improved;
        self.spec_alpha_distinct_winners = report.distinct_winners;
        self.spec_alpha_incumbent_grad_improvements = report.incumbent_grad_improvements;
        self.spec_alpha_closed_extra = report.closed_extra;
        self.spec_alpha_truncated = report.truncated;
        self.spec_alpha_root_before_bits = Some(report.root_before.to_bits());
        self.spec_alpha_root_after_bits = Some(report.root_after.to_bits());
        self.spec_alpha_best_gain_bits = Some(report.best_gain.to_bits());
        self
    }
}

impl RootAlphaExecutionEvidence {
    /// The owned direct root proof completed before any alpha/rebake owner
    /// was dispatched. Preserve the configured policy, and report no attempt
    /// rather than interpreting enabled configuration as execution evidence.
    pub(crate) fn skipped_by_direct_cover(
        policy: super::RootAlphaRuntimePolicy,
        eligible: bool,
    ) -> Self {
        let mut evidence = Self::completed(policy, eligible, None, None);
        evidence.standalone_attempted = 0;
        evidence.rebake_attempted = 0;
        evidence.standalone_suppressed = 0;
        evidence.rebake_suppressed = 0;
        evidence
    }

    pub(crate) fn completed(
        policy: super::RootAlphaRuntimePolicy,
        eligible: bool,
        standalone: Option<&AlphaOptReport>,
        rebake: Option<&super::rebake::RebakeReport>,
    ) -> Self {
        let standalone_matches = standalone.is_none_or(|report| {
            policy.alpha_enabled && eligible && report.cfg == AlphaOptCfg::from_policy(policy)
        });
        let rebake_matches = rebake
            .is_none_or(|report| policy.rebake_enabled && eligible && report.policy == policy);
        Self {
            alpha_enabled: policy.alpha_enabled,
            alpha_iters: policy.alpha_iters,
            alpha_secs: policy.alpha_secs,
            alpha_lr: policy.alpha_lr,
            alpha_cols: policy.alpha_cols,
            alpha_guard: policy.alpha_guard,
            rebake_enabled: policy.rebake_enabled,
            rebake_rounds: policy.rebake_rounds,
            rebake_secs: policy.rebake_secs,
            spec_alpha_enabled: policy.spec_alpha_enabled,
            spec_alpha_width: policy.spec_alpha_width,
            spec_alpha_cands: policy.spec_alpha_cands,
            spec_alpha_rounds: policy.spec_alpha_rounds,
            spec_alpha_entries: policy.spec_alpha_entries,
            // YIELD: "did not run" until `with_spec_alpha_yield` says otherwise.
            spec_alpha_ran: false,
            spec_alpha_rounds_run: 0,
            spec_alpha_passes: 0,
            spec_alpha_entries_committed: 0,
            spec_alpha_classes_selected: 0,
            spec_alpha_incumbent_grads_ok: 0,
            spec_alpha_classes_improved: 0,
            spec_alpha_distinct_winners: 0,
            spec_alpha_incumbent_grad_improvements: 0,
            spec_alpha_closed_extra: 0,
            spec_alpha_truncated: 0,
            spec_alpha_root_before_bits: None,
            spec_alpha_root_after_bits: None,
            spec_alpha_best_gain_bits: None,
            eligible,
            standalone_suppressed: usize::from(eligible && !policy.alpha_enabled),
            standalone_attempted: usize::from(policy.alpha_enabled),
            standalone_completed: usize::from(standalone.is_some()),
            standalone_effective_secs: standalone.map_or(0.0, |report| report.effective_secs),
            standalone_deadline_capped: standalone.is_some_and(|report| report.deadline_capped),
            standalone_budget_exhausted: standalone.is_some_and(|report| report.budget_exhausted),
            standalone_iters_cap_exhausted: standalone
                .is_some_and(|report| report.iters_cap_exhausted),
            standalone_fail_classes: standalone.map_or(0, |report| report.fail_classes),
            standalone_iters_attempted: standalone.map_or(0, |report| report.iters),
            standalone_candidate_steps: standalone.map_or(0, |report| report.candidate_steps),
            standalone_legacy_steps: standalone.map_or(0, |report| report.legacy_steps),
            standalone_adam_steps: standalone.map_or(0, |report| report.adam_steps),
            standalone_passes: standalone.map_or(0, |report| report.passes),
            standalone_max_steer_cols: standalone.map_or(0, |report| report.max_steer_cols),
            standalone_frozen_improvements: standalone
                .map_or(0, |report| report.frozen_improvements),
            standalone_accepted: standalone.is_some_and(|report| report.accepted),
            standalone_moved_neurons: standalone.map_or(0, |report| report.moved_neurons),
            standalone_guard_opportunities: standalone
                .map_or(0, |report| report.guard_opportunities),
            standalone_guard_baseline_evals: standalone
                .map_or(0, |report| report.guard_baseline_evals),
            standalone_guard_candidate_evals: standalone
                .map_or(0, |report| report.guard_candidate_evals),
            standalone_guard_reversions: standalone.map_or(0, |report| report.guard_reversions),
            standalone_frozen_before_bits: standalone.map(|report| report.baseline.to_bits()),
            standalone_frozen_best_bits: standalone.map(|report| report.best.to_bits()),
            standalone_published_before_bits: standalone
                .and_then(|report| report.published_before.map(f64::to_bits)),
            standalone_published_after_bits: standalone
                .and_then(|report| report.published_after.map(f64::to_bits)),
            rebake_suppressed: usize::from(eligible && !policy.rebake_enabled),
            rebake_attempted: usize::from(policy.rebake_enabled),
            rebake_completed: usize::from(rebake.is_some()),
            rebake_effective_secs: rebake.map_or(0.0, |report| report.effective_secs),
            rebake_deadline_capped: rebake.is_some_and(|report| report.deadline_capped),
            rebake_budget_exhausted: rebake.is_some_and(|report| report.budget_exhausted),
            rebake_round_cap_exhausted: rebake.is_some_and(|report| report.round_cap_exhausted),
            rebake_entry_unbaked_alpha: rebake.is_some_and(|report| report.entry_unbaked_alpha),
            rebake_baseline_evals: usize::from(rebake.is_some()),
            rebake_rounds_started: rebake.map_or(0, |report| report.rounds_started),
            rebake_alpha_completed: rebake.map_or(0, |report| report.alpha_completed),
            rebake_alpha_iters_attempted: rebake.map_or(0, |report| report.alpha_iters_attempted),
            rebake_alpha_candidate_steps: rebake.map_or(0, |report| report.alpha_candidate_steps),
            rebake_alpha_legacy_steps: rebake.map_or(0, |report| report.alpha_legacy_steps),
            rebake_alpha_adam_steps: rebake.map_or(0, |report| report.alpha_adam_steps),
            rebake_alpha_passes: rebake.map_or(0, |report| report.alpha_passes),
            rebake_alpha_max_fail_classes: rebake.map_or(0, |report| report.alpha_max_fail_classes),
            rebake_alpha_max_steer_cols: rebake.map_or(0, |report| report.alpha_max_steer_cols),
            rebake_alpha_moved_neurons: rebake.map_or(0, |report| report.alpha_moved_neurons),
            rebake_rebuilds_completed: rebake.map_or(0, |report| report.rebuilds_completed),
            rebake_rounds_scored: rebake.map_or(0, |report| report.rounds_run),
            rebake_rounds_accepted: rebake.map_or(0, |report| report.rounds_accepted),
            rebake_fixed_point_stops: rebake.map_or(0, |report| report.fixed_point_stops),
            rebake_affordability_refusals: rebake.map_or(0, |report| report.affordability_refusals),
            rebake_aborts: rebake.map_or(0, |report| report.aborts),
            rebake_published_before_bits: rebake.map(|report| report.baseline.to_bits()),
            rebake_published_after_bits: rebake.map(|report| report.best.to_bits()),
            policy_mismatch: !(standalone_matches && rebake_matches),
        }
    }
}

impl Default for RootAlphaExecutionEvidence {
    fn default() -> Self {
        Self::completed(super::RootAlphaRuntimePolicy::default(), false, None, None)
    }
}

/// Frozen-policy entry: no-op unless root alpha is armed.
///
/// Called from `lane_impl` between the root build and the BaB driver, where
/// the root is still owned mutably. Bound-quality only: an internal error
/// stops further candidates; only the best earlier, strictly improving
/// certified candidate can still be committed.
pub fn maybe_optimize_root_alpha(
    net: &TwinNet,
    root: &mut RootGates,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    policy: super::RootAlphaRuntimePolicy,
) -> Option<AlphaOptReport> {
    if !policy.alpha_enabled {
        return None;
    }
    let mut cfg = AlphaOptCfg::from_policy(policy);
    // #router-r3 `rule.ia_yield`: the standalone ascent reads a CONTRACT. Any
    // interm time the yielded candidate loop freed is excluded from its window
    // (consumed here, once), so it reaches the tree. ZERO unless the acting
    // optimizer armed `rule.ia_yield` and the interm phase ended early.
    cfg.contract_credit = ny_core::stop_authority::global().take_ascent_credit();
    optimize_root_alpha(net, root, t, adv, deadline, cfg)
}

/// The optimizer proper (test-visible; the policy gate lives in
/// [`maybe_optimize_root_alpha`]).
pub(crate) fn optimize_root_alpha(
    net: &TwinNet,
    root: &mut RootGates,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    cfg: AlphaOptCfg,
) -> Option<AlphaOptReport> {
    // #phase-cost: the root alpha ascent is an alpha optimizer run.
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::AlphaOpt);
    let t0 = Instant::now();
    if !root.mode.outward() {
        // Parity is the bit-parity oracle against core.py; it must not move.
        eprintln!("[alpha-opt] refuse: parity mode");
        return None;
    }
    if adv.is_empty() || root.layers.iter().all(|l| l.unst.is_empty()) {
        eprintln!(
            "[alpha-opt] refuse: nothing to optimize (no adv classes or no unstable neurons)"
        );
        return None;
    }
    // #alpha-frac: cap the ascent at a FRACTION of the remaining deadline.
    //
    // Historically a hardcoded 0.4, with the stated purpose "never starve the
    // tree search". MEASURED 2026-09-16 that guard is aimed at the wrong phase:
    // alpha-beta-CROWN's ENTIRE branch-and-bound is 0-11% of its solve
    // (0.000s / 0.184s / 0.444s on three probe rows), both verifiers spend their
    // budget at the ROOT, and every tree-directed mechanism measured that day
    // converted nothing. Meanwhile abc's root alpha ascent buys +0.1218 of bound
    // to ny's +0.0328 — the residual that decides ~22 of ny's 50 cifar100 gap
    // rows.
    //
    // It also made `margin_row.alpha_secs` INERT: with ~48s remaining the
    // fraction yields 19.2s and always wins the `min`, so raising `alpha_secs`
    // to 34.0 changed the spend by 0.26s and the root bound by ZERO.
    //
    // ny already has a typed key for exactly this — `root_alpha_frac`
    // (`ny-cli/src/preset/mod.rs:2072`) — but it is wired to the GRAPH lane
    // (`beta_crown/config/beta_config.rs:991`) and occurs zero times here, while
    // cifar100 proves through THIS lane. That is the 13th wrong-lane instance.
    //
    // `ALPHA_REMAINING_FRAC` ships at 0.4, byte-identical to the literal it
    // replaces.
    //
    // #router-r3 `rule.ia_yield`: the share is taken of the window MINUS the
    // contract credit (interm time the yielded candidates freed), so the
    // ascent can only get less than today's rule would give it, never more.
    // A ZERO credit (every run the acting optimizer does not touch) is
    // today's expression exactly.
    let budget = match deadline {
        Some(dl) => cfg.secs.min(
            ALPHA_REMAINING_FRAC
                * dl.saturating_duration_since(t0)
                    .saturating_sub(cfg.contract_credit)
                    .as_secs_f64(),
        ),
        None => cfg.secs,
    };
    if !cfg.contract_credit.is_zero() {
        let today = deadline.map_or(cfg.secs, |dl| {
            cfg.secs
                .min(ALPHA_REMAINING_FRAC * dl.saturating_duration_since(t0).as_secs_f64())
        });
        eprintln!(
            "[optimizer] contract rule=rule.ia_yield site=margin-row.alpha credit_ms={} \
             budget_s={budget:.3} today_s={today:.3}",
            cfg.contract_credit.as_millis()
        );
    }
    let deadline_capped = budget < cfg.secs;
    if budget < 0.5 {
        eprintln!("[alpha-opt] refuse: no budget (deadline too close)");
        return None;
    }
    // #root-eval-reuse (throughput plan B1(c)): `certified_root_bound` STORES
    // this evaluation on `root`. The frozen frame below borrows its y-rows, and
    // if the guard rolls the iterate back, the tree's own root evaluation of
    // these same gates (`bab::run_inner`) is served from it.
    let published_before = if cfg.guard {
        match super::rebake::certified_root_bound(net, root, t, adv) {
            Ok(bound) if bound.is_finite() => Some(bound),
            Ok(_) => {
                eprintln!("[alpha-opt] refuse: non-finite published baseline");
                return None;
            }
            Err(error) => {
                eprintln!("[alpha-opt] refuse: published baseline root_eval: {error}");
                return None;
            }
        }
    } else {
        None
    };
    let sr = search_alpha(net, &*root, t, adv, t0, budget, cfg, CERTIFIED_TAG)?;
    // COMMIT: strictly-better certified objective only (monotone accept). The
    // heuristic alpha is otherwise retained bit-identically.
    let mut accepted = sr.best > sr.baseline;
    let mut moved_neurons = 0usize;
    let mut max_dalpha = 0.0f64;
    // A rollback copy is needed only when the published-frame guard may reject
    // an otherwise accepted frozen-frame iterate. Avoid cloning every layer on
    // the common no-improvement path (and when the guard is explicitly off).
    let original = (accepted && published_before.is_some()).then(|| {
        root.layers
            .iter()
            .map(|layer| (layer.alpha.clone(), layer.ms.clone()))
            .collect::<Vec<_>>()
    });
    // #head-alpha: the head slopes roll back with the trunk's.
    let original_head = root.head_alpha.clone();
    let mut head_committed = false;
    let guard_opportunities = usize::from(accepted);
    let guard_candidate_evals = usize::from(accepted && published_before.is_some());
    let mut guard_reversions = 0usize;
    let mut published_after = None;
    let mut saturated_alphas = 0usize;
    let mut committed_alphas = 0usize;
    if accepted {
        for (li, layer) in root.layers.iter_mut().enumerate() {
            let cand = &sr.best_alpha[li];
            // Touch ONLY unstable neurons: stable and split-baked neurons keep
            // their exact fixed lines bit-identically.
            let unst = layer.unst.clone();
            for j in unst {
                let a_new = cand[j].clamp(0.0, 1.0);
                let a_old = layer.alpha[j];
                // #alpha-saturation PC-2: is the committed value ON the clamp?
                committed_alphas += 1;
                if a_new == 0.0 || a_new == 1.0 {
                    saturated_alphas += 1;
                }
                if a_new != a_old {
                    moved_neurons += 1;
                    max_dalpha = max_dalpha.max((a_new - a_old).abs());
                    layer.alpha[j] = a_new;
                }
                // Re-derive the alpha-coupled certified quantities via the same
                // path the root build uses:
                //  * `s`/`c` are functions of `(l, u)` ONLY (`gates_from_box` +
                //    `repair_upper_lines`), and `(l, u)` are frozen here, so
                //    their build-time certificate stands verbatim — nothing to
                //    recompute.
                //  * `ms = max(alpha, s)` (root.rs) IS alpha-coupled: it is the
                //    error-contraction Lipschitz constant of the backward gate
                //    transform, and a stale value under a raised alpha would
                //    under-scale the carried error (false-UNSAT risk). Recompute.
                layer.ms[j] = layer.alpha[j].max(layer.s[j]);
            }
        }
        // #head-alpha: install the ascended head slopes. `root_eval` (the
        // published guard below, and every tree domain) reads them through
        // `head_gates_with`, only where a head neuron is unstable.
        if let Some(h) = sr.best_head.as_ref() {
            if sr.head_moved > 0 {
                root.head_alpha = Some(h.clone());
                head_committed = true;
                // `moved_neurons` / `max_dalpha` count every committed slope,
                // head included, so "accepted" always implies "something moved".
                moved_neurons += sr.head_moved;
                max_dalpha = max_dalpha.max(sr.head_max_delta);
            }
        }
    }
    if accepted {
        if let Some(before) = published_before {
            // #root-eval-reuse (B1(c)): stored as well — when the guard KEEPS
            // the iterate, the tree's root evaluation of these gates reuses it.
            let after = super::rebake::certified_root_bound(net, root, t, adv);
            let keep = after
                .as_ref()
                .is_ok_and(|value| value.is_finite() && *value > before);
            published_after = after
                .as_ref()
                .ok()
                .copied()
                .filter(|value| value.is_finite());
            match after {
                Ok(value) => eprintln!(
                    "[alpha-opt] published_guard before={before:.6} after={value:.6} accepted={keep}"
                ),
                Err(error) => eprintln!(
                    "[alpha-opt] published_guard before={before:.6} after=error accepted=false: {error}"
                ),
            }
            if !keep {
                let original = original.expect("accepted guarded iterate retained rollback state");
                for (layer, (alpha, ms)) in root.layers.iter_mut().zip(original) {
                    layer.alpha = alpha;
                    layer.ms = ms;
                }
                root.head_alpha = original_head;
                head_committed = false;
                accepted = false;
                moved_neurons = 0;
                max_dalpha = 0.0;
                guard_reversions = 1;
            }
        }
    }
    // #alpha-saturation PC-2, LIFTED out of the `if accepted { if let
    // Some(before) = published_before {` nest it used to live in, and moved
    // here so it describes the alpha this root actually PUBLISHES (after any
    // guard rollback) rather than only an accepted-and-guarded iterate.
    //
    // Nested, the line printed only when the guard was ON *and* a frozen-frame
    // iterate had been accepted, so a run that never accepted produced no line
    // and its null was uninterpretable — absence of the line could not be told
    // from absence of saturation. Unconditional here: absence of this line now
    // means the binary lacks the counter.
    //
    // READ IT WITH `accepted=` IN HAND. When nothing was committed the alpha is
    // still the BINARY area heuristic, so `frac=1.0` is the NULL, not a finding.
    // The measurement that bears on a richer alpha axis is the fraction under
    // `accepted=true`.
    {
        let mut published_committed = 0usize;
        let mut published_at_clamp = 0usize;
        for layer in &root.layers {
            for &j in &layer.unst {
                published_committed += 1;
                if layer.alpha[j] == 0.0 || layer.alpha[j] == 1.0 {
                    published_at_clamp += 1;
                }
            }
        }
        if published_committed > 0 {
            eprintln!(
                "[alpha-opt] saturation committed={published_committed} at_clamp={published_at_clamp} \
frac={:.4} accepted={accepted} scanned_on_commit={committed_alphas} at_clamp_on_commit={saturated_alphas}",
                published_at_clamp as f64 / published_committed as f64
            );
        }
    }
    eprintln!(
        "[alpha-opt] head-alpha commit: moved={} committed={head_committed}",
        sr.head_moved
    );
    eprintln!(
        "[alpha-opt] done iters={} passes={} root_bound_before={:.6} after={:.6} accepted={} \
moved_neurons={} max_dalpha={:.4} lr={:.4} cols={} guard={} secs={:.2}",
        sr.iters,
        sr.passes,
        sr.baseline,
        sr.best,
        accepted,
        moved_neurons,
        max_dalpha,
        cfg.lr,
        cfg.cols,
        cfg.guard,
        t0.elapsed().as_secs_f64()
    );
    Some(AlphaOptReport {
        cfg,
        effective_secs: budget,
        deadline_capped,
        budget_exhausted: sr.budget_exhausted,
        iters_cap_exhausted: sr.candidate_steps == cfg.iters,
        max_steer_cols: sr.max_steer_cols,
        guard_baseline_evals: usize::from(published_before.is_some()),
        guard_candidate_evals,
        guard_reversions,
        fail_classes: sr.fail_classes,
        candidate_steps: sr.candidate_steps,
        legacy_steps: if cfg.lr == 0.0 { sr.candidate_steps } else { 0 },
        adam_steps: if cfg.lr == 0.0 { 0 } else { sr.candidate_steps },
        frozen_improvements: sr.frozen_improvements,
        guard_opportunities,
        saturated_alphas,
        committed_alphas,
        published_before,
        published_after,
        baseline: sr.baseline,
        best: sr.best,
        iters: sr.iters,
        passes: sr.passes,
        accepted,
        moved_neurons,
        max_dalpha,
        governor_yield_iter: sr.governor_yield.map(|y| y.iter),
        governor_freed_secs: sr.governor_yield.map_or(0.0, |y| y.freed_secs),
        contract_credit_secs: cfg.contract_credit.as_secs_f64(),
    })
}

/// #router-r3 `rule.governor`: where the ascent's own yield rule stopped it.
#[derive(Debug, Clone, Copy)]
struct GovernorYield {
    iter: usize,
    freed_secs: f64,
}

struct SearchRes {
    /// #head-alpha: the best head slopes (`None` when no head coordinate
    /// existed), and how many unstable head neurons they moved.
    best_head: Option<Vec<f64>>,
    head_moved: usize,
    head_max_delta: f64,
    best_alpha: Vec<Vec<f64>>,
    baseline: f64,
    best: f64,
    iters: usize,
    passes: usize,
    candidate_steps: usize,
    budget_exhausted: bool,
    fail_classes: usize,
    max_steer_cols: usize,
    frozen_improvements: usize,
    /// #router-r3: the governor's yield, when it fired.
    governor_yield: Option<GovernorYield>,
}

/// Log tag of the CERTIFIED root ascent: every historical `[alpha-opt]` line,
/// byte-identical.
pub(crate) const CERTIFIED_TAG: &str = "[alpha-opt]";

/// #root-certify-once: log tag of the UNTRUSTED ascent. Distinct on purpose —
/// its `iter=` lines report Parity ranking numbers, and a reader who greps
/// `[alpha-opt] iter=` must never count them as certified ascent iterations.
pub(crate) const UNTRUSTED_TAG: &str = "[alpha-opt:untrusted]";

/// #root-certify-once: what one UNTRUSTED ascent did. Search telemetry only —
/// nothing here is a bound, and nothing here may reach a verdict.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UntrustedAscent {
    /// Candidate iterations attempted (excludes the baseline pass).
    pub(crate) iters: usize,
    /// Parity backward passes spent (baseline + candidates).
    pub(crate) passes: usize,
    /// Parity frozen-frame objective of the entry alpha (a ranking number).
    pub(crate) baseline: f64,
    /// Best Parity frozen-frame objective reached (a ranking number).
    pub(crate) best: f64,
    /// Scratch trunk + head slopes the commit moved.
    pub(crate) moved: usize,
    /// The ascent stopped because the next pass did not fit its budget.
    pub(crate) budget_exhausted: bool,
}

/// #root-certify-once: the root alpha ascent run UNCERTIFIED on a Parity
/// SCRATCH root, committing its best iterate into that scratch.
///
/// This is `search_alpha` — the very same frozen-frame objective,
/// supergradient, Adam step, head-alpha coordinates and monotone accept the
/// certified ascent uses — evaluated on a root whose `mode` is
/// [`RoundMode::Parity`](super::rounding::RoundMode): every pass is plain f64
/// round-to-nearest with no `|W|` error lane (the census's
/// `outward_over_parity` is ~2.0, so a pass costs about half of a certified
/// one), and every number it reports RANKS alphas and is never a bound.
///
/// # Why this cannot reach a verdict
///
/// * It REFUSES an Outward root: the only thing it may ever mutate is a Parity
///   scratch, which no verdict path reads (`bab::closed` on a Parity number is
///   non-strict precisely because such a number is not a certificate).
/// * The alphas it commits into the scratch are only PROPOSALS. The caller
///   (`root_certify_once`) installs them into the certified root through a
///   finite/`[0, 1]` ingest gate with `ms = max(alpha, s)` re-derived, and
///   publishes them only if ONE certified Outward `root_eval` of the result
///   strictly beats the certified incumbent. Any `alpha in [0, 1]` is a valid
///   DeepPoly lower line (`root.rs`), so a mis-ranked proposal costs tightness,
///   never validity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn untrusted_ascent(
    net: &TwinNet,
    scratch: &mut RootGates,
    t: usize,
    adv: &[usize],
    budget: f64,
    iters: usize,
    lr: f64,
    cols: usize,
) -> Option<UntrustedAscent> {
    if scratch.mode.outward() {
        eprintln!(
            "{UNTRUSTED_TAG} refuse: the untrusted ascent runs only on a Parity scratch root, \
never on a certified one"
        );
        return None;
    }
    if iters == 0 || !(budget.is_finite() && budget > 0.0) {
        return None;
    }
    if adv.is_empty() || scratch.layers.iter().all(|layer| layer.unst.is_empty()) {
        eprintln!(
            "{UNTRUSTED_TAG} refuse: nothing to optimize (no adv classes or no unstable neurons)"
        );
        return None;
    }
    let t0 = Instant::now();
    let cfg = AlphaOptCfg {
        iters,
        secs: budget,
        lr: if lr.is_finite() {
            lr.clamp(0.0, 1.0)
        } else {
            0.0
        },
        cols: cols.max(1),
        guard: false,
        governor: None,
        contract_credit: std::time::Duration::ZERO,
    };
    let sr = search_alpha(net, &*scratch, t, adv, t0, budget, cfg, UNTRUSTED_TAG)?;
    let mut moved = 0usize;
    if sr.best > sr.baseline {
        for (li, layer) in scratch.layers.iter_mut().enumerate() {
            let Some(cand) = sr.best_alpha.get(li) else {
                continue;
            };
            for pos in 0..layer.unst.len() {
                let j = layer.unst[pos];
                let Some(&proposed) = cand.get(j) else {
                    continue;
                };
                if !proposed.is_finite() {
                    continue;
                }
                let a = proposed.clamp(0.0, 1.0);
                if a != layer.alpha[j] {
                    layer.alpha[j] = a;
                    layer.ms[j] = a.max(layer.s[j]);
                    moved += 1;
                }
            }
        }
        if let Some(head) = sr.best_head {
            if sr.head_moved > 0 && head.iter().all(|a| a.is_finite()) {
                scratch.head_alpha = Some(head.iter().map(|a| a.clamp(0.0, 1.0)).collect());
                moved += sr.head_moved;
            }
        }
    }
    Some(UntrustedAscent {
        iters: sr.iters,
        passes: sr.passes,
        baseline: sr.baseline,
        best: sr.best,
        moved,
        budget_exhausted: sr.budget_exhausted,
    })
}

/// The search: immutable root, candidates injected as `DomainGates`.
///
/// `tag` prefixes every line this search prints. The certified ascent passes
/// `CERTIFIED_TAG` (`[alpha-opt]`, byte-identical to the historical lines);
/// `#root-certify-once` passes `UNTRUSTED_TAG`, so an UNTRUSTED search's
/// `iter=` lines can never be read as certified ones. Every number the search
/// computes is in `root.mode`: on a Parity scratch root the whole frozen frame
/// (y-rows, head gates, margin seeds, via-y composition, direct pass,
/// concretization) is Parity and is a ranking number, never a bound.
#[allow(clippy::too_many_arguments)]
fn search_alpha(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
    t0: Instant,
    budget: f64,
    cfg: AlphaOptCfg,
    tag: &'static str,
) -> Option<SearchRes> {
    let eng = BackwardEngine::new(net, root);
    let mode = root.mode;
    if let Ok(mut st) = JOINT_STATS.lock() {
        *st = (0, 0.0, 0.0);
    }
    // ---- Frozen evaluation frame (built ONCE under the heuristic gates) ----
    // y-pack -> y-box -> head gates -> margin seeds -> m1 / m2v. All stay
    // fixed while trunk alpha moves; every quantity is a valid certified bound
    // for ANY valid trunk relaxation, so freezing costs tightness of the
    // OBJECTIVE only, never soundness. The published root bound is recomputed
    // from scratch by `root_eval` under the committed gates.
    let mb_all = match MarginBatch::new(net, t, adv) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{tag} abort: margin batch: {e}");
            return None;
        }
    };
    // #root-eval-reuse (throughput plan B1(c)): the published-frame guard's
    // baseline (`certified_root_bound`, above in `optimize_root_alpha`) — or a
    // rebake round's evaluation — already ran exactly this y-row pair on these
    // exact gates; borrow its pack's rows when the gates generation matches.
    // A miss is the unchanged `y_rows` call.
    let memo_rows = super::bab::root_rows_memo(net, root);
    let owned_rows;
    let (al, au): (&PassOut, &PassOut) = match &memo_rows {
        Some(rows) => (rows.al(), rows.au()),
        None => {
            // #mr-census: this frozen-frame y-row refresh is the same 2 x n_y
            // pass pair `build_pack` runs, so it is counted under `y_rows`.
            let census_rows =
                super::census::SiteTimer::start(super::census::Site::YRows, 2 * net.n_y);
            owned_rows = match eng.y_rows(None) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{tag} abort: y-pack: {e}");
                    return None;
                }
            };
            drop(census_rows);
            (&owned_rows.0, &owned_rows.1)
        }
    };
    let ybox = YBox::from_rows(&eng, al, au);
    let gates = head_gates_with(&ybox, mode, root.head_alpha.as_deref());
    let ms_all = margin_seed(&mb_all, &gates, &ybox, mode);
    let al_dots = row_dots(root, al);
    let au_dots = row_dots(root, au);
    let m2v_all = compose_viay(&eng, &mb_all, &gates, al, au, &al_dots, &au_dots, mode);
    // Mirror root_eval's admission: only classes the frozen head paths do NOT
    // already close get a direct-pass column (same width as the root's own
    // direct pass; outward closure is strict `> 0`).
    let fail: Vec<usize> = (0..adv.len())
        .filter(|&k| {
            let d = ms_all.m1[k].max(m2v_all[k]);
            !(d.is_finite() && d > 0.0)
        })
        .collect();
    if fail.is_empty() {
        eprintln!("{tag} refuse: all classes close under the frozen head frame");
        return None;
    }
    let fail_classes: Vec<usize> = fail.iter().map(|&k| adv[k]).collect();
    let mbf = match MarginBatch::new(net, t, &fail_classes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{tag} abort: fail-set margin batch: {e}");
            return None;
        }
    };
    let ms_f = margin_seed(&mbf, &gates, &ybox, mode);
    let m2v_f: Vec<f64> = fail.iter().map(|&k| m2v_all[k]).collect();
    let nf = mbf.nf();
    // #head-alpha: the head neurons UNSTABLE in the frozen y-box are the head
    // coordinates of the ascent. Every other head neuron keeps its exact line.
    let head_unst: Vec<usize> = if HEAD_ALPHA {
        (0..ybox.ly.len())
            .filter(|&j| ybox.ly[j] < 0.0 && ybox.uy[j] > 0.0)
            .collect()
    } else {
        Vec::new()
    };
    let head_on = !head_unst.is_empty();
    let head_init: Vec<f64> = gates.alpha.clone();

    // One evaluation = one certified Lower pass under the candidate alpha
    // (with the pre-transform coefficient capture the gradient needs) plus the
    // frozen-frame objective: min over classes of max(direct, m1, m2v).
    // #head-alpha: with head coordinates, the seed and the via-y composition
    // are rebuilt under the candidate head slopes (both are cheap next to the
    // pass); `m1` is the interval head path and does not depend on them.
    // Without head coordinates the frozen `ms_f` / `m2v_f` are used verbatim.
    let eval = |alpha: &[Vec<f64>],
                head: &[f64]|
     -> Result<(f64, Vec<usize>, usize, PassOut), String> {
        let head_frame = head_on.then(|| {
            let hg = HeadGates {
                alpha: head.to_vec(),
                s: gates.s.clone(),
                c: gates.c.clone(),
            };
            let ms = margin_seed(&mbf, &hg, &ybox, mode);
            let m2v = compose_viay(&eng, &mbf, &hg, al, au, &al_dots, &au_dots, mode);
            (ms, m2v)
        });
        let (ms_use, m2v_use): (&MarginSeed, &[f64]) = match &head_frame {
            Some((ms, m2v)) => (ms, m2v.as_slice()),
            None => (&ms_f, m2v_f.as_slice()),
        };
        let dom = gates_override(root, alpha);
        // #mr-census: the root ascent's certified evaluation pass.
        let census_pass =
            super::census::SiteTimer::start(super::census::Site::RootAscent, ms_use.seed.s.ncols());
        let pass = eng
            .run_collect(
                &ms_use.seed,
                Some(&dom),
                LaneDir::Lower,
                None,
                Collect {
                    unst_abs: false,
                    rows: None,
                    unst_rows: true,
                },
            )
            .map_err(|e| e.to_string())?;
        drop(census_pass);
        let direct = per_class_direct(&eng, &pass, ms_use, 0..nf);
        let mut scored = Vec::with_capacity(nf);
        for f in 0..nf {
            // #steer-branch: `frozen` is the part of the objective alpha CANNOT
            // move. `direct >= frozen` means the max is attained by the branch
            // `supergradient` actually differentiates; a tie counts as active,
            // since `d(direct)/d(alpha)` is then a valid supergradient of `d`.
            let frozen = ms_use.m1[f].max(m2v_use[f]);
            let d = direct[f].max(frozen);
            if !d.is_finite() {
                return Err("non-finite objective".into());
            }
            scored.push((d, f, direct[f] >= frozen));
        }
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let worst = scored[0].0;
        let (steer_cols, pinned) =
            steer_selection(&scored, cfg.cols.min(nf).max(1), STEER_ACTIVE_BRANCH);
        Ok((worst, steer_cols, pinned, pass))
    };

    let init: Vec<Vec<f64>> = root.layers.iter().map(|l| l.alpha.clone()).collect();
    let te = Instant::now();
    let (baseline, mut steer_cols, pinned0, baseline_pass) = match eval(&init, &head_init) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{tag} abort: baseline pass: {e}");
            return None;
        }
    };
    let mut pass_secs = te.elapsed().as_secs_f64().max(1e-3);
    // #steer-branch's OWN observable, emitted once per ascent. `frozen_pinned`
    // is the count of open classes whose objective a frozen term attains, i.e.
    // whose true supergradient in alpha is ZERO. A run reporting
    // `frozen_pinned=0` is a clean "the defect is not present on this row",
    // not a VOID.
    eprintln!(
        "{tag} steer-branch: armed={STEER_ACTIVE_BRANCH} open_classes={nf} \
direct_active={} frozen_pinned={pinned0} steer_cols={}",
        nf - pinned0,
        steer_cols.len()
    );
    if STEER_ACTIVE_BRANCH && steer_cols.is_empty() {
        eprintln!(
            "{tag} stop=frozen_pinned: every one of the {nf} open classes has its \
objective attained by a frozen term, so the objective is locally constant in alpha and no \
iterate can move it; returning {:.1}s of the {budget:.1}s ascent budget to the tree search",
            (budget - t0.elapsed().as_secs_f64()).max(0.0)
        );
    }
    eprintln!(
        "{tag} iter=0 root_bound_before={baseline:.6} after={baseline:.6} accepted=true \
(baseline; frozen-frame objective over {nf} classes)"
    );
    let mut best = baseline;
    let mut best_alpha = init;
    // #mr-census: the root ascent's supergradient walks (one per steer column).
    let census_walk =
        super::census::SiteTimer::start(super::census::Site::RootWalk, steer_cols.len());
    let mut grad = mean_supergradient(net, root, &best_alpha, &baseline_pass, &steer_cols);
    // #head-alpha: head slopes, their supergradient, and their Adam moments.
    let mut best_head = head_init.clone();
    let mut hgrad = if head_on {
        head_supergradient(
            net,
            root,
            &best_alpha,
            &baseline_pass,
            &steer_cols,
            &mbf,
            &head_unst,
        )
    } else {
        None
    };
    drop(census_walk);
    let mut head_m = vec![0.0f64; best_head.len()];
    let mut head_v = head_m.clone();
    let mut eta = 0.25f64;
    let mut adam_m: Vec<Vec<f64>> = root.layers.iter().map(|l| vec![0.0; l.n]).collect();
    let mut adam_v = adam_m.clone();
    // #channel-alpha: per-trunk-relu spatial group size (H*W), and the count of
    // steps the projection actually altered — the mechanism's own observable.
    let chan_groups = channel_group_sizes(net);
    let mut chan_projected = 0usize;
    let mut passes = 1usize;
    let mut iters = 0usize;
    let mut candidate_steps = 0usize;
    let mut budget_exhausted = false;
    let mut max_steer_cols = steer_cols.len();
    let mut frozen_improvements = 0usize;
    // #budget-governor: the ascent's yield gate. Disarmed (`GOVERNOR_RATIO ==
    // None`) it never fires and the loop is byte-identical to today.
    // #router-r3 `rule.governor`: the acting optimizer's stop authority arms
    // it (0.15 after 5 samples, the moat-cleared artifact); unarmed it is the
    // constant above, so every other run is byte-identical.
    let (governor_ratio, governor_min, governor_source) =
        super::marginal::ascent_governor(cfg.governor);
    let mut yield_gate = MarginalYield::new(governor_ratio, governor_min);
    let mut governor_yield: Option<GovernorYield> = None;
    // #router-r6 `rule.trial_lr_backoff`: the Adam step's BACKOFF machine
    // (`None` unless the acting optimizer armed the site, and never on the
    // legacy `lr == 0` step, which already halves its own eta on every
    // rejection). Unarmed `lr_now == cfg.lr` exactly, so every step below is
    // today's arithmetic bit for bit.
    let mut lr_trial = if cfg.lr == 0.0 {
        None
    } else {
        crate::trial_sites::lr_backoff(crate::trial_sites::global())
    };
    for it in 1..=cfg.iters {
        // Leave room for the pass we are about to spend.
        if t0.elapsed().as_secs_f64() + 1.5 * pass_secs > budget {
            budget_exhausted = true;
            break;
        }
        let lr_now = lr_trial.as_ref().map_or(cfg.lr, |t| cfg.lr * t.scale());
        let Some(g) = grad.as_ref() else { break };
        let gmax = g
            .iter()
            .flat_map(|v| v.iter())
            .chain(hgrad.iter().flat_map(|h| h.iter()))
            .fold(0.0f64, |m, &x| m.max(x.abs()));
        if !(gmax.is_finite() && gmax > 0.0) {
            break;
        }
        // Projected step FROM THE BEST iterate. `lr == 0` is the exact legacy
        // globally-normalized update; otherwise Adam makes the step scale-free
        // per coordinate, with alpha-beta-CROWN's 0.98 learning-rate decay.
        let mut cand = best_alpha.clone();
        let mut moved = false;
        for (li, layer) in root.layers.iter().enumerate() {
            // #channel-alpha: materialise this layer's steps first so they can be
            // projected onto the channel-constant subspace before they are
            // applied. OFF => the vector is consumed exactly as the historical
            // in-place loop did, in the same order, so the arithmetic is
            // unchanged.
            let mut steps = vec![0.0f64; cand[li].len()];
            for &j in &layer.unst {
                let gn = g[li][j] / gmax;
                steps[j] = if cfg.lr == 0.0 {
                    eta * gn
                } else {
                    adam_m[li][j] = 0.9 * adam_m[li][j] + 0.1 * gn;
                    adam_v[li][j] = 0.999 * adam_v[li][j] + 0.001 * gn * gn;
                    let mhat = adam_m[li][j] / (1.0 - 0.9f64.powi(it as i32));
                    let vhat = adam_v[li][j] / (1.0 - 0.999f64.powi(it as i32));
                    let lr = lr_now * 0.98f64.powi((it - 1) as i32);
                    lr * mhat / (vhat.sqrt() + 1e-8)
                };
            }
            if CHANNEL_ALPHA {
                if let Some(Some(group)) = chan_groups.get(li) {
                    let n = project_step_to_channels(&mut steps, &layer.unst, *group);
                    chan_projected += n;
                }
            }
            for &j in &layer.unst {
                let a2 = (cand[li][j] + steps[j]).clamp(0.0, 1.0);
                if a2 != cand[li][j] {
                    cand[li][j] = a2;
                    moved = true;
                }
            }
        }
        // #head-alpha: the same projected step on the head coordinates, from
        // the best head iterate, with its own Adam moments.
        let mut cand_head = best_head.clone();
        if let Some(hg) = hgrad.as_ref() {
            for &j in &head_unst {
                let gn = hg[j] / gmax;
                let step = if cfg.lr == 0.0 {
                    eta * gn
                } else {
                    head_m[j] = 0.9 * head_m[j] + 0.1 * gn;
                    head_v[j] = 0.999 * head_v[j] + 0.001 * gn * gn;
                    let mhat = head_m[j] / (1.0 - 0.9f64.powi(it as i32));
                    let vhat = head_v[j] / (1.0 - 0.999f64.powi(it as i32));
                    let lr = lr_now * 0.98f64.powi((it - 1) as i32);
                    lr * mhat / (vhat.sqrt() + 1e-8)
                };
                let a2 = (cand_head[j] + step).clamp(0.0, 1.0);
                if a2 != cand_head[j] {
                    cand_head[j] = a2;
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
        if CHANNEL_ALPHA && it == 1 {
            // #channel-alpha's OWN observable, emitted once per ascent: a run
            // that projected ZERO steps is INERT, and any measurement from it is
            // VOID rather than negative.
            eprintln!(
                "{tag} channel-alpha: projected_steps={chan_projected} conv_layers={}",
                chan_groups.iter().filter(|g| g.is_some()).count()
            );
        }
        iters = it;
        let te = Instant::now();
        let iterate_t0 = te;
        let (f, fcols, _pinned, pass) = match eval(&cand, &cand_head) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{tag} abort: pass failed at iter {it}: {e}");
                break;
            }
        };
        pass_secs = pass_secs.max(te.elapsed().as_secs_f64());
        passes += 1;
        candidate_steps += 1;
        max_steer_cols = max_steer_cols.max(fcols.len());
        let improved = f > best;
        eprintln!(
            "{tag} iter={it} root_bound_before={best:.6} after={f:.6} accepted={improved} \
step={} cols={}",
            if cfg.lr == 0.0 {
                format!("legacy_eta:{eta:.3}")
            } else {
                format!("adam_lr:{:.4}", lr_now * 0.98f64.powi((it - 1) as i32))
            },
            steer_cols.len()
        );
        // #ascent-marginal-stop: measure the gain BEFORE `best` moves.
        let marginal_delta = f - best;
        // #router-r6 `rule.trial_lr_backoff`: the iteration boundary is the
        // ascent's safe point. The machine sees whether the phase's own
        // acceptance KEPT the iterate and its gain over the best so far; the
        // best iterate stays the committed one whatever the step.
        if lr_trial.is_some() {
            let done_frac = (it as f64 / cfg.iters.max(1) as f64)
                .max(t0.elapsed().as_secs_f64() / budget.max(f64::MIN_POSITIVE));
            crate::trial_sites::observe_lr(
                crate::trial_sites::global(),
                &mut lr_trial,
                ny_core::router::trial::Iterate {
                    index: it as u64,
                    accepted: improved,
                    gain: marginal_delta,
                    secs: iterate_t0.elapsed().as_secs_f64(),
                    done_frac,
                },
            );
        }
        // #router-r1: write-only per-iteration gain tap (the `rule.governor`
        // shadow input). Nothing reads it back; one relaxed load while off.
        if ny_core::router::bus::enabled() {
            ny_core::router::bus::emit(ny_core::router::bus::ProgressSample::scalar(
                ny_core::router::decl::ids::MARGIN_ROW_ALPHA,
                it as u64,
                marginal_delta,
            ));
        }
        if improved {
            frozen_improvements += 1;
            best = f;
            best_alpha = cand;
            best_head = cand_head;
            steer_cols = fcols;
            let census_walk =
                super::census::SiteTimer::start(super::census::Site::RootWalk, steer_cols.len());
            grad = mean_supergradient(net, root, &best_alpha, &pass, &steer_cols);
            if head_on {
                hgrad = head_supergradient(
                    net,
                    root,
                    &best_alpha,
                    &pass,
                    &steer_cols,
                    &mbf,
                    &head_unst,
                );
            }
            drop(census_walk);
        } else if cfg.lr == 0.0 {
            eta *= 0.5;
            if eta < 0.01 {
                break;
            }
        }
        // #ascent-marginal-stop: the ascent has converged; every further
        // iteration costs the TREE ~0.65 s to buy ~1e-5 of root bound. Returning
        // the remainder is what converts these rows.
        // #budget-governor: this phase has converged relative to its own best
        // rate; every further iteration costs the TREE ~0.65 s to buy ~1e-5 of
        // root bound. Returning the remainder is what converts these rows.
        if yield_gate.observe(marginal_delta) {
            let used_secs = t0.elapsed().as_secs_f64();
            let freed_secs = (budget - used_secs).max(0.0);
            eprintln!(
                "{tag} governor-yield at iter={it}: delta={marginal_delta:.3e} < \
floor={:.3e} (ratio x best={:.3e}); returning {:.1}s of the {budget:.1}s ascent \
budget to the tree search",
                yield_gate.floor(),
                yield_gate.best(),
                freed_secs
            );
            governor_yield = Some(GovernorYield {
                iter: it,
                freed_secs,
            });
            if governor_source == GovernorSource::Router {
                // #router-r3 observable + ledger: the best iterate stays the
                // committed one (`best_alpha`) and is certified ONCE by the
                // caller's unchanged published guard; the freed seconds are
                // handed to the tree at its start (`[optimizer] hand-on`).
                let secs_ms = |secs: f64| (secs.max(0.0) * 1e3).min(u64::MAX as f64) as u64;
                eprintln!(
                    "[optimizer] yield rule=rule.governor site=margin-row.alpha tag={tag} \
                     iter={it} budget_ms={} used_ms={} freed_ms={} rate={marginal_delta:.3e} \
                     best={:.3e} floor={:.3e} keep=best_iterate",
                    secs_ms(budget),
                    secs_ms(used_secs),
                    secs_ms(freed_secs),
                    yield_gate.best(),
                    yield_gate.floor()
                );
                ny_core::stop_authority::global().record_yield(
                    ny_core::stop_authority::YieldEvent {
                        site: ny_core::stop_authority::StopSite::AlphaAscent,
                        tag: tag.to_owned(),
                        unit: it as u64,
                        budget_ms: secs_ms(budget),
                        used_ms: secs_ms(used_secs),
                        freed_ms: secs_ms(freed_secs),
                        rate: marginal_delta,
                        best: yield_gate.best(),
                        floor: yield_gate.floor(),
                    },
                );
            }
            iters = it;
            break;
        }
    }
    // #joint-grad's OWN observable. `ratio` is the joint term's magnitude
    // against the direct term it augments: armed with `ratio=0` the mechanism
    // is INERT and any measurement from it is VOID, not negative.
    if let Ok(st) = JOINT_STATS.lock() {
        let (calls, d, j) = *st;
        eprintln!(
            "{tag} joint-grad: armed={JOINT_GRAD} probe={JOINT_GRAD_PROBE} refreshes={calls} sum_direct={d:.6e} \
sum_joint={j:.6e} ratio={:.6}",
            if d > 0.0 { j / d } else { 0.0 }
        );
    }
    // #head-alpha's OWN observable, emitted once per ascent: `unstable=0` or
    // `armed=false` means the head coordinates never existed, so a run printing
    // either is VOID for this mechanism, not negative.
    let head_moved = head_unst
        .iter()
        .filter(|&&j| best_head[j] != head_init[j])
        .count();
    let head_max_delta = head_unst
        .iter()
        .fold(0.0f64, |m, &j| m.max((best_head[j] - head_init[j]).abs()));
    let head_frac = if head_unst.is_empty() {
        0.0
    } else {
        head_unst
            .iter()
            .filter(|&&j| best_head[j] > 0.0 && best_head[j] < 1.0)
            .count() as f64
            / head_unst.len() as f64
    };
    eprintln!(
        "{tag} head-alpha: armed={HEAD_ALPHA} unstable={} moved={head_moved} \
interior_frac={head_frac:.4} grad={}",
        head_unst.len(),
        hgrad.is_some()
    );
    Some(SearchRes {
        best_head: head_on.then_some(best_head),
        head_moved,
        head_max_delta,
        best_alpha,
        baseline,
        best,
        iters,
        passes,
        candidate_steps,
        budget_exhausted,
        fail_classes: nf,
        max_steer_cols,
        frozen_improvements,
        governor_yield,
    })
}

/// `#joint-grad`, measured 2026-09-17: ONE sweep for the BINDING column, not one
/// per column.
///
/// The probe build folded the joint term inside `supergradient`, so the reverse
/// sweep ran once per COLUMN per refresh — 96 sweeps on `idx_1280` — and that
/// overhead alone cut the ascent from 20 iterations to 15. The term is worth
/// `ratio` 0.058-0.101 of the direct gradient, which cannot pay for a 25%
/// iteration loss. So the correction is now computed for `cols[0]` only.
///
/// That is also the RIGHT column on principle, not merely the cheap one: the
/// objective is `min_k d_k`, whose supergradient is the ATTAINING class's, and
/// `steer_cols` is sorted ascending by `d` — so `cols[0]` IS the argmin class.
/// The mean over K is itself only a proxy for that one gradient.
fn joint_supergradient(
    net: &TwinNet,
    root: &RootGates,
    alpha: &[Vec<f64>],
    pass: &PassOut,
    col: usize,
) -> Option<Vec<Vec<f64>>> {
    // Same prologue as `supergradient`: the concretized argmin of the Lower
    // bound for this column.
    let vsigns = pass.unst_rows.as_ref()?;
    let n_in = root.mid.len();
    let r = pass.a.ncols();
    if col >= r {
        return None;
    }
    let asl = pass.a.as_slice()?;
    let mut xstar = vec![0.0; n_in];
    for i in 0..n_in {
        let a = asl[i * r + col];
        xstar[i] = if a > 0.0 {
            root.mid[i] - root.rad[i]
        } else if a < 0.0 {
            root.mid[i] + root.rad[i]
        } else {
            root.mid[i]
        };
    }
    let (u, _y) = linearized_walk(net, root, alpha, vsigns, col, &xstar, false)?;
    joint_term(net, root, alpha, vsigns, &u, col)
}

/// Average the frozen-sign supergradients of the K worst objective columns.
/// K=1 deliberately calls the historical primitive once and is therefore the
/// exact steering control arm.
fn mean_supergradient(
    net: &TwinNet,
    root: &RootGates,
    alpha: &[Vec<f64>],
    pass: &PassOut,
    cols: &[usize],
) -> Option<Vec<Vec<f64>>> {
    if cols.is_empty() {
        return None;
    }
    // The K columns are INDEPENDENT work: `supergradient` borrows
    // `net`/`root`/`alpha`/`pass` immutably, allocates its own output, and
    // nothing beneath it carries interior mutability or shared scratch (its one
    // already-parallel callee, `conv_apply_forward`, hands every worker its own
    // accumulator tile and keeps the per-output tap order fixed).
    //
    // WHAT IT COSTS, counted rather than assumed. This function has exactly two
    // call sites: once at baseline (:824) and once per ACCEPTED iterate (:905,
    // inside `if improved {`). A rejected iterate re-uses the previous `grad`
    // and pays nothing. So an ascent pays `1 + frozen_improvements` of these
    // K-walk batches, NOT one per iteration — worth stating precisely, because
    // the loop is deadline-bound and the obvious wrong reading (one per
    // iteration) overstates the prize.
    //
    // BIT-IDENTICAL PER CALL. The parallelism is confined to PRODUCING the
    // gradients. `collect` here is over an indexed parallel iterator, so
    // `grads[i]` is always the gradient of `cols[i]` whichever worker computed
    // it, and the accumulation below then runs SERIALLY over that vector — the
    // same addends into `mean` in the same column order as the original serial
    // loop. f64 addition is not associative, so the ORDER, not merely the set
    // of addends, is what makes every `mean[li][j]` byte-for-byte the value
    // this function returned before. A `par_iter().reduce`/`fold` would
    // reassociate that sum and is deliberately not used.
    //
    // DELIBERATELY *NOT* BIT-IDENTICAL END-TO-END, and this must not be
    // misread as a no-op. The ascent's stop rule (:836) is WALL-CLOCK —
    // `t0.elapsed() + 1.5 * pass_secs > budget` — so a cheaper gradient buys
    // MORE iterations. `best_alpha`, the published root bound, `iters`,
    // `passes`, `frozen_improvements` and every `[alpha-opt] iter=` line will
    // therefore move on real rows. That is the intended win, not a regression,
    // and it means an A/B against a banked run is NOT expected to be
    // byte-identical.
    //
    // It is nonetheless SOUND, for reasons that live outside this function:
    // every iterate is re-scored by the unchanged certified `eval` before
    // `improved = f > best` (:889), and under `cfg.guard` the whole ascent is
    // reverted unless the INDEPENDENTLY recomputed published root beats
    // `published_before` (:558, checked at :621). More iterations can only
    // offer more candidates to those two gates; neither gate is relaxed here.
    //
    // Refusal semantics are unchanged too. The serial loop returned `None` at
    // the first column whose walk refused; here every column is walked and the
    // `?` below fires at that same index. `supergradient` is pure, so the extra
    // walks past a refusal are unobservable.
    let grads: Vec<Option<Vec<Vec<f64>>>> = cols
        .par_iter()
        .map(|&col| supergradient(net, root, alpha, pass, col))
        .collect();
    // ---- #alpha-spec-probe. THE PRE-REGISTERED KILL GATE for #spec-alpha.
    //
    // This function is where the per-start-node information ny already pays for
    // is DESTROYED: `supergradient` returns the per-(neuron, column) quantity
    // `vp * U` for each of the K steer columns, and the accumulate below
    // averages them into a single [li][j] direction. If every column wants the
    // same thing, the shared alpha already IS each column's own optimum and a
    // per-start-node axis buys nothing. If they disagree, the average is
    // destroying a real signal.
    //
    // Measured BEFORE the accumulate, from `grads`, so it costs one extra pass
    // over data already in hand and NO extra certified pass, NO extra wall
    // budget, and no key. Print-only; `mean` is byte-identical either way.
    //
    // READING RULE, pre-registered:
    //   KILL  if frac < 0.05 AND cos_mean >= 0.8 AND argmin_switches <= 1
    //   FUND  if frac >= 0.05 OR cos_min < 0 OR argmin_switches >= 3
    // UNCONDITIONAL, deliberately. A raw env gate here would be a new
    // undeclared `NY_*` read against `ny-levers/tests/ratchet.rs`, and a typed
    // preset key would be absurd for a print-only counter. Unconditional also
    // means absence of the line is evidence the BINARY lacks it, rather than
    // evidence the run declined — which is the whole point of the observable.
    // Cost is O(K * unstable) per call against a certified backward pass.
    if cols.len() > 1 {
        spec_disagreement_probe(root, &grads, cols.len());
    }
    let mut mean: Vec<Vec<f64>> = root.layers.iter().map(|l| vec![0.0; l.n]).collect();
    for grad in grads {
        let grad = grad?;
        for (dst_layer, src_layer) in mean.iter_mut().zip(grad) {
            for (dst, src) in dst_layer.iter_mut().zip(src_layer) {
                *dst += src;
            }
        }
    }
    let scale = 1.0 / cols.len() as f64;
    for layer in &mut mean {
        for value in layer {
            *value *= scale;
        }
    }
    // #joint-grad: the term the frozen frame omits, added ONCE at full weight
    // for the binding column. Full weight — not `scale`d — because `mean` is a
    // proxy for the argmin class's gradient and this is that class's missing
    // term, not a K-th contribution to an average. Disarmed it only measures.
    if JOINT_GRAD || JOINT_GRAD_PROBE {
        if let Some(&col0) = cols.first() {
            if let Some(gj) = joint_supergradient(net, root, alpha, pass, col0) {
                let (mut nd, mut nj) = (0.0f64, 0.0f64);
                for (li, gl) in gj.iter().enumerate() {
                    let Some(mrow) = mean.get(li) else { continue };
                    for (j, &v) in gl.iter().enumerate() {
                        if let Some(&d) = mrow.get(j) {
                            nd += d.abs();
                            nj += v.abs();
                        }
                    }
                }
                if let Ok(mut st) = JOINT_STATS.lock() {
                    st.0 += 1;
                    st.1 += nd;
                    st.2 += nj;
                }
                if JOINT_GRAD {
                    for (li, gl) in gj.iter().enumerate() {
                        let Some(mrow) = mean.get_mut(li) else {
                            continue;
                        };
                        for (j, &v) in gl.iter().enumerate() {
                            if let Some(d) = mrow.get_mut(j) {
                                *d += v;
                            }
                        }
                    }
                }
            }
        }
    }
    Some(mean)
}

/// #alpha-spec-probe: how much do the K steer columns DISAGREE about alpha?
///
/// Print-only and unconditional whenever the steering width exceeds one. It
/// changes no bound: `mean` is byte-identical whether or not this runs.
///
/// Reported over the unstable neurons with support in at least one column:
///  * `sign_disagree` / `frac` — neurons where two columns' gradients have
///    strictly opposite signs. This is the quantity that decides whether one
///    shared slope can be optimal for every class at once.
///  * `cos_mean` / `cos_min` — pairwise cosine between per-column gradient
///    vectors. `cos_min < 0` is a hard disagreement.
///  * `max_spread` — largest per-neuron range of the normalized gradient.
fn spec_disagreement_probe(root: &RootGates, grads: &[Option<Vec<Vec<f64>>>], k: usize) {
    let mut vecs: Vec<Vec<f64>> = Vec::with_capacity(k);
    for grad in grads.iter().flatten() {
        let mut flat: Vec<f64> = Vec::new();
        for (li, layer) in root.layers.iter().enumerate() {
            for &j in &layer.unst {
                flat.push(grad[li][j]);
            }
        }
        vecs.push(flat);
    }
    if vecs.len() < 2 || vecs[0].is_empty() {
        eprintln!("[alpha-spec-probe] refuse: fewer than two usable column gradients");
        return;
    }
    let n = vecs[0].len();
    let mut support = 0usize;
    let mut disagree = 0usize;
    let mut max_spread = 0.0f64;
    // Per-column L2 norms, for the spread normalization and the cosines.
    let norms: Vec<f64> = vecs
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt())
        .collect();
    for i in 0..n {
        let mut pos = false;
        let mut neg = false;
        let mut any = false;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for (c, v) in vecs.iter().enumerate() {
            let g = v[i];
            if g != 0.0 {
                any = true;
                if g > 0.0 {
                    pos = true;
                } else {
                    neg = true;
                }
            }
            if norms[c] > 0.0 {
                let scaled = g / norms[c];
                lo = lo.min(scaled);
                hi = hi.max(scaled);
            }
        }
        if any {
            support += 1;
            if pos && neg {
                disagree += 1;
            }
            if hi > lo {
                max_spread = max_spread.max(hi - lo);
            }
        }
    }
    let mut cos_sum = 0.0f64;
    let mut cos_min = f64::INFINITY;
    let mut pairs = 0usize;
    for a in 0..vecs.len() {
        for b in (a + 1)..vecs.len() {
            if norms[a] == 0.0 || norms[b] == 0.0 {
                continue;
            }
            let dot: f64 = vecs[a].iter().zip(&vecs[b]).map(|(x, y)| x * y).sum();
            let cos = dot / (norms[a] * norms[b]);
            cos_sum += cos;
            cos_min = cos_min.min(cos);
            pairs += 1;
        }
    }
    // How often does the per-column ARGMAX neuron (the one the column would
    // move hardest) change between columns? >= 3 distinct winners is the
    // pre-registered FUND signal.
    let mut winners: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for v in &vecs {
        let mut best = (0usize, f64::NEG_INFINITY);
        for (i, g) in v.iter().enumerate() {
            if g.abs() > best.1 {
                best = (i, g.abs());
            }
        }
        winners.insert(best.0);
    }
    let frac = if support > 0 {
        disagree as f64 / support as f64
    } else {
        0.0
    };
    let cos_mean = if pairs > 0 {
        cos_sum / pairs as f64
    } else {
        f64::NAN
    };
    eprintln!(
        "[alpha-spec-probe] cols={} support={support} sign_disagree={disagree} frac={frac:.4} \
cos_mean={cos_mean:.4} cos_min={:.4} max_spread={max_spread:.4} argmin_switches={} unstable={n}",
        vecs.len(),
        if pairs > 0 { cos_min } else { f64::NAN },
        winners.len()
    );
}

/// Full-layer gate override for a candidate alpha. `s`/`c` are the root's
/// certified chords (untouched); `ms` is re-derived from the CANDIDATE alpha —
/// the one alpha-coupled certified quantity (see the module doc).
/// #head-alpha: supergradient of the steer columns' direct objective w.r.t.
/// the head slopes, averaged over `cols` in column order (the same serial
/// accumulation contract as [`mean_supergradient`]).
///
/// `g[j] = mean_f wp[f, j] * y_f,j(x*_f)`, where `y_f(x*_f)` is the frozen-sign
/// linearized walk's head pre-activation at the column's concretization argmin
/// (`linearized_walk(.., head = true)`). Only `head_unst` entries are filled.
/// Direction only; nothing verdict-bearing reads it.
fn head_supergradient(
    net: &TwinNet,
    root: &RootGates,
    alpha: &[Vec<f64>],
    pass: &PassOut,
    cols: &[usize],
    mb: &MarginBatch,
    head_unst: &[usize],
) -> Option<Vec<f64>> {
    if cols.is_empty() {
        return None;
    }
    let vsigns = pass.unst_rows.as_ref()?;
    let n_in = root.mid.len();
    let r = pass.a.ncols();
    let n_y = mb.n_y;
    let asl = pass.a.as_slice()?;
    let ys: Vec<Option<Vec<f64>>> = cols
        .par_iter()
        .map(|&col| {
            if col >= r || (col + 1) * n_y > mb.wp.len() {
                return None;
            }
            let xstar: Vec<f64> = (0..n_in)
                .map(|i| {
                    let a = asl[i * r + col];
                    if a > 0.0 {
                        root.mid[i] - root.rad[i]
                    } else if a < 0.0 {
                        root.mid[i] + root.rad[i]
                    } else {
                        root.mid[i]
                    }
                })
                .collect();
            let (_u, y) = linearized_walk(net, root, alpha, vsigns, col, &xstar, true)?;
            (y.len() == n_y).then_some(y)
        })
        .collect();
    let mut g = vec![0.0f64; n_y];
    for (&col, y) in cols.iter().zip(ys) {
        let y = y?;
        let wp = &mb.wp[col * n_y..(col + 1) * n_y];
        for &j in head_unst {
            g[j] += wp[j] * y[j];
        }
    }
    let scale = 1.0 / cols.len() as f64;
    for v in &mut g {
        *v *= scale;
    }
    Some(g)
}

fn gates_override(root: &RootGates, alpha: &[Vec<f64>]) -> DomainGates {
    let mut dg = DomainGates::default();
    for (li, layer) in root.layers.iter().enumerate() {
        if layer.unst.is_empty() {
            continue;
        }
        let ms: Vec<f64> = alpha[li]
            .iter()
            .zip(&layer.s)
            .map(|(a, s)| a.max(*s))
            .collect();
        dg.layers.insert(
            li,
            GateVecs {
                alpha: alpha[li].clone(),
                s: layer.s.clone(),
                c: layer.c.clone(),
                ms,
            },
        );
    }
    dg
}

/// Supergradient of the objective column `col` w.r.t. every unstable trunk
/// alpha: `g[li][j] = vp_{li,j,col} * U_{li,j,col}` (see the module doc).
///
/// Heuristic-only: the direction never feeds a verdict — every accepted
/// iterate is re-scored by the unchanged certified pass.
///
/// `pub(crate)`: #margin-row-domain-alpha reuses this EXACT gradient per BaB
/// domain, passing the domain's effective alpha vectors — the same
/// direction-only contract `beta::step_scales` already relies on. Note the
/// gradient is NOT sign-definite: `vp >= 0` but `U` is the pre-activation of an
/// UNSTABLE neuron at `x*`, so it straddles zero. (The banked "ny's alpha
/// gradient is sign-definite <= 0" finding is about the INTERNAL verifier's
/// alpha-CROWN, a different lane; it does not apply here.)
/// `#joint-grad` accounting, so an armed run that produced NO joint signal is
/// distinguishable from one that produced a useless one. `(refreshes, sum of
/// mean |direct|, sum of mean |joint|)`; reset at the top of every ascent.
pub(crate) static JOINT_STATS: std::sync::Mutex<(u64, f64, f64)> =
    std::sync::Mutex::new((0, 0.0, 0.0));

/// Build the joint frame for one column and run the reverse sweep.
///
/// Everything here is per-column data `supergradient` already holds: the frozen
/// -sign walk values `u`, the accumulated coefficients `vsigns`, and the layer
/// boxes. `box_weights` converts each unstable neuron's coefficient into the
/// sensitivity of the bound to that neuron's own `(l, u)`; `joint_adjoint`
/// carries those weights back to every EARLIER layer's alpha.
fn joint_term(
    net: &TwinNet,
    root: &RootGates,
    alpha: &[Vec<f64>],
    vsigns: &BTreeMap<usize, Array2<f64>>,
    u: &BTreeMap<usize, Vec<f64>>,
    col: usize,
) -> Option<Vec<Vec<f64>>> {
    let nl = root.layers.len();
    let (mut al, mut sl, mut unst) = (
        Vec::with_capacity(nl),
        Vec::with_capacity(nl),
        Vec::with_capacity(nl),
    );
    let (mut seed, mut upper, mut zl) = (
        Vec::with_capacity(nl),
        Vec::with_capacity(nl),
        Vec::with_capacity(nl),
    );
    for (li, layer) in root.layers.iter().enumerate() {
        al.push(alpha.get(li)?.clone());
        sl.push(layer.s.clone());
        let k = layer.unst.len();
        let (mut sd, mut up, mut zz) = (vec![0.0; k], vec![false; k], vec![0.0; k]);
        if let (Some(uvals), Some(vmat)) = (u.get(&li), vsigns.get(&li)) {
            let vs = vmat.as_slice()?;
            let rr = vmat.ncols();
            if uvals.len() != k || vmat.nrows() != k || col >= rr {
                return None;
            }
            for (pos, &j) in layer.unst.iter().enumerate() {
                let v = vs[pos * rr + col];
                let z = uvals[pos];
                zz[pos] = z;
                up[pos] = v < 0.0;
                let (dl, du) = box_weights(v, z, *layer.l.get(j)?, *layer.u.get(j)?);
                // `l` and `u` are both bounds on the same pre-activation `z`;
                // evaluating both derivatives at the single walk point makes
                // their contributions share one reverse path. See the module
                // docs — this is the mechanism's one stated approximation.
                sd[pos] = dl + du;
            }
        }
        unst.push(layer.unst.clone());
        seed.push(sd);
        upper.push(up);
        zl.push(zz);
    }
    joint_adjoint(
        net,
        &JointFrame {
            alpha: &al,
            s: &sl,
            unst: &unst,
            seed: &seed,
            upper: &upper,
            z: &zl,
        },
    )
}

pub(crate) fn supergradient(
    net: &TwinNet,
    root: &RootGates,
    alpha: &[Vec<f64>],
    pass: &PassOut,
    col: usize,
) -> Option<Vec<Vec<f64>>> {
    let vsigns = pass.unst_rows.as_ref()?;
    // Concretize argmin of the Lower bound for this column:
    // x*_i = mid_i - sign(A_i)*rad_i (mid on a zero coefficient).
    let n_in = root.mid.len();
    let r = pass.a.ncols();
    if col >= r {
        return None;
    }
    let asl = pass.a.as_slice()?;
    let mut xstar = vec![0.0; n_in];
    for i in 0..n_in {
        let a = asl[i * r + col];
        xstar[i] = if a > 0.0 {
            root.mid[i] - root.rad[i]
        } else if a < 0.0 {
            root.mid[i] + root.rad[i]
        } else {
            root.mid[i]
        };
    }
    let (u, _y) = linearized_walk(net, root, alpha, vsigns, col, &xstar, false)?;
    let mut g: Vec<Vec<f64>> = root.layers.iter().map(|l| vec![0.0; l.n]).collect();
    for (li, layer) in root.layers.iter().enumerate() {
        let (Some(uvals), Some(vmat)) = (u.get(&li), vsigns.get(&li)) else {
            continue;
        };
        let vs = vmat.as_slice()?;
        let rr = vmat.ncols();
        if uvals.len() != layer.unst.len() || vmat.nrows() != layer.unst.len() || col >= rr {
            return None;
        }
        for (pos, &j) in layer.unst.iter().enumerate() {
            let vp = vs[pos * rr + col].max(0.0);
            g[li][j] = vp * uvals[pos];
        }
    }
    Some(g)
}

/// Forward walk of `x*` through the FROZEN-SIGN linearized trunk: every op is
/// applied exactly (f64, with bias); every relu is replaced per neuron by the
/// affine line the pass selected for column `col` (Lower lane: `alpha*z` on a
/// positive incoming coefficient, `s*z + c` on a negative one; exact fixed
/// lines — stable and split-baked neurons — are sign-independent).
///
/// Returns `(U, y)`: per trunk layer, the walk values at that layer's UNSTABLE
/// neurons captured BEFORE the layer's own transform — the sensitivities `U` —
/// and, when `head` is set, the head pre-activation vector `ŷ(x*)` (the first
/// head Gemm's output; empty when `head` is false).
///
/// `head = false` reproduces the original walk BIT-IDENTICALLY (break at the
/// last trunk relu, no head ops touched). `head = true` (#margin-row-beta C3)
/// additionally applies the LAST trunk relu's selected lines — the same code
/// path as interior relus — and continues through the remaining trunk-tail ops
/// to `net.i_gemm1`, applying the head Gemm there. A Gemm anywhere OTHER than
/// exactly `i_gemm1` still refuses (`None`); relus between the last trunk relu
/// and `i_gemm1` cannot exist (net.rs compile: every relu before gemm1 is a
/// trunk relu).
///
/// `pub(crate)`: #margin-row-beta reuses this walk for its per-split
/// supergradient (`beta::step_scales`) — same frozen-sign linearization, same
/// direction-only contract (nothing verdict-bearing reads the result).
pub(crate) fn linearized_walk(
    net: &TwinNet,
    root: &RootGates,
    alpha: &[Vec<f64>],
    vsigns: &BTreeMap<usize, Array2<f64>>,
    col: usize,
    xstar: &[f64],
    head: bool,
) -> Option<(BTreeMap<usize, Vec<f64>>, Vec<f64>)> {
    let last_relu = *net.trunk_relus.last()?;
    let stop = if head { net.i_gemm1 } else { last_relu };
    let mut vals: Vec<Option<Vec<f64>>> = vec![None; net.ops.len() + 1];
    vals[0] = Some(xstar.to_vec());
    let mut u_out: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for (k, op) in net.ops.iter().enumerate().take(stop + 1) {
        let out: Vec<f64> = match op {
            TwinOp::Conv(c) => {
                let src = vals[c.input].as_ref()?;
                let n_out = net.tsize[k + 1];
                let src_m = Array2::from_shape_vec((src.len(), 1), src.clone()).ok()?;
                let mut dst = Array2::<f64>::zeros((n_out, 1));
                conv_apply_forward(c, &src_m, &mut dst, false);
                let p = c.oshape.1 * c.oshape.2;
                let mut v = dst.as_slice()?.to_vec();
                for (j, vj) in v.iter_mut().enumerate() {
                    *vj += c.bias[j / p];
                }
                v
            }
            TwinOp::Add { lhs, rhs } => {
                let a = vals[*lhs].as_ref()?;
                let b = vals[*rhs].as_ref()?;
                a.iter().zip(b).map(|(x, y)| x + y).collect()
            }
            TwinOp::Flatten { input } => vals[*input].as_ref()?.clone(),
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                ..
            } => {
                let src = vals[*input].as_ref()?;
                src.iter()
                    .enumerate()
                    .map(|(j, &v)| scale[j] * v + shift[j])
                    .collect()
            }
            TwinOp::Relu { input, layer } => {
                let li = *layer;
                let src = vals[*input].as_ref()?;
                let rec = root.layers.get(li)?;
                if src.len() != rec.n {
                    return None;
                }
                // Sensitivities: pre-transform values at the unstable neurons.
                if !rec.unst.is_empty() {
                    u_out.insert(li, rec.unst.iter().map(|&j| src[j]).collect());
                }
                if k == last_relu && !head {
                    break;
                }
                // Default per-neuron map: the (candidate) lower line `a*z`.
                // For every exact-line neuron — stable active (1,1,0), stable
                // inactive (0,0,0), split-baked — this IS the selected map
                // (candidates only move unstable alphas). Unstable neurons are
                // overwritten below with the pass's per-column selection.
                let mut out: Vec<f64> = src
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| alpha[li][j] * v)
                    .collect();
                if let Some(vmat) = vsigns.get(&li) {
                    let vs = vmat.as_slice()?;
                    let rr = vmat.ncols();
                    if vmat.nrows() != rec.unst.len() || col >= rr {
                        return None;
                    }
                    for (pos, &j) in rec.unst.iter().enumerate() {
                        // Negative incoming coefficient -> the pass selected
                        // the upper chord for this (neuron, column). A zero
                        // coefficient contributes nothing to the bound either
                        // way; keep the lower line (matches `vp = 0`).
                        if vs[pos * rr + col] < 0.0 {
                            out[j] = rec.s[j] * src[j] + rec.c[j];
                        }
                    }
                }
                out
            }
            // Direction-only walk: a BN-folded head is evaluated at its STORED
            // weights (its certified fold error only matters to bounds, and
            // every bound this direction feeds is re-scored by the charged
            // certified pass).
            TwinOp::Gemm {
                input,
                weight,
                bias,
                shape,
                ..
            } => {
                // #margin-row-beta C3: the FIRST head Gemm maps the trunk
                // output to the pre-head-relu vector y — exactly the layer the
                // head supergradient needs. Allowed only at exactly `i_gemm1`
                // in head mode; any other Gemm refuses as before.
                if !(head && k == net.i_gemm1) {
                    return None; // structurally pre-gemm otherwise
                }
                let src = vals[*input].as_ref()?;
                let (no, ni) = *shape;
                if src.len() != ni {
                    return None;
                }
                (0..no)
                    .map(|j| {
                        let row = &weight[j * ni..(j + 1) * ni];
                        row.iter().zip(src).map(|(w, v)| w * v).sum::<f64>() + bias[j]
                    })
                    .collect()
            }
        };
        vals[k + 1] = Some(out);
    }
    let y = if head {
        vals[net.i_gemm1 + 1].take()?
    } else {
        Vec::new()
    };
    Some((u_out, y))
}

#[cfg(test)]
mod execution_evidence_tests {
    use super::{AlphaOptCfg, AlphaOptReport, RootAlphaExecutionEvidence};
    use crate::margin_row::{rebake::RebakeReport, RootAlphaRuntimePolicy};

    #[test]
    fn root_alpha_evidence_is_route_split_and_policy_bound() {
        let policy = RootAlphaRuntimePolicy {
            alpha_enabled: true,
            alpha_iters: 4,
            alpha_secs: 12.5,
            alpha_lr: 0.05,
            alpha_cols: 3,
            alpha_guard: true,
            rebake_enabled: true,
            rebake_rounds: 2,
            rebake_secs: 30.0,
            spec_alpha_enabled: true,
            spec_alpha_width: 2,
            spec_alpha_cands: 4,
            spec_alpha_rounds: 2,
            spec_alpha_entries: 4_096,
        };
        let standalone = AlphaOptReport {
            cfg: AlphaOptCfg::from_policy(policy),
            effective_secs: 10.0,
            deadline_capped: true,
            budget_exhausted: false,
            iters_cap_exhausted: true,
            max_steer_cols: 3,
            guard_baseline_evals: 1,
            guard_candidate_evals: 1,
            guard_reversions: 0,
            fail_classes: 7,
            candidate_steps: 4,
            legacy_steps: 0,
            adam_steps: 4,
            frozen_improvements: 1,
            guard_opportunities: 1,
            saturated_alphas: 0,
            committed_alphas: 0,
            published_before: Some(-1.0),
            published_after: Some(-0.5),
            baseline: -1.25,
            best: -0.75,
            iters: 4,
            passes: 5,
            accepted: true,
            moved_neurons: 6,
            max_dalpha: 0.25,
            governor_yield_iter: None,
            governor_freed_secs: 0.0,
            contract_credit_secs: 0.0,
        };
        let rebake = RebakeReport {
            policy,
            effective_secs: 20.0,
            deadline_capped: true,
            budget_exhausted: true,
            round_cap_exhausted: false,
            entry_unbaked_alpha: true,
            rounds_started: 1,
            alpha_completed: 1,
            alpha_iters_attempted: 4,
            alpha_candidate_steps: 4,
            alpha_legacy_steps: 0,
            alpha_adam_steps: 4,
            alpha_passes: 5,
            alpha_max_fail_classes: 7,
            alpha_max_steer_cols: 3,
            alpha_moved_neurons: 6,
            rebuilds_completed: 1,
            fixed_point_stops: 0,
            affordability_refusals: 1,
            aborts: 0,
            baseline: -1.0,
            best: -0.25,
            rounds_run: 1,
            rounds_accepted: 1,
        };

        let evidence =
            RootAlphaExecutionEvidence::completed(policy, true, Some(&standalone), Some(&rebake));
        assert!(!evidence.policy_mismatch);
        assert_eq!(evidence.standalone_candidate_steps, 4);
        assert_eq!(
            evidence.standalone_frozen_before_bits,
            Some((-1.25f64).to_bits())
        );
        assert_eq!(
            evidence.standalone_published_after_bits,
            Some((-0.5f64).to_bits())
        );
        assert_eq!(evidence.rebake_alpha_candidate_steps, 4);
        assert_eq!(evidence.rebake_rounds_scored, 1);
        assert_eq!(
            evidence.rebake_published_after_bits,
            Some((-0.25f64).to_bits())
        );

        let suppressed = RootAlphaExecutionEvidence::completed(
            RootAlphaRuntimePolicy::default(),
            true,
            None,
            None,
        );
        assert_eq!(suppressed.standalone_suppressed, 1);
        assert_eq!(suppressed.rebake_suppressed, 1);

        let disabled_report = RootAlphaExecutionEvidence::completed(
            RootAlphaRuntimePolicy {
                alpha_enabled: false,
                ..policy
            },
            true,
            Some(&standalone),
            None,
        );
        assert!(disabled_report.policy_mismatch);
    }
}

#[cfg(test)]
mod alpha_frac_tests {
    use super::ALPHA_REMAINING_FRAC;

    /// The budget expression as `optimize` computes it.
    fn budget(cfg_secs: f64, remaining: f64, frac: f64) -> f64 {
        cfg_secs.min(frac * remaining)
    }

    /// THE MEASURED INERTNESS this parameterisation exists to explain
    /// (2026-09-16, cifar100 resnet_large idx_1280_sidx_5310):
    /// arming `alpha_secs: 34.0` moved the spend 19.49s -> 19.23s and the root
    /// bound not at all, because the FRACTION binds, not `cfg.secs`.
    #[test]
    fn the_fraction_binds_not_alpha_secs() {
        let remaining = 48.0; // measured: ~48s left when the ascent starts
        let shipped = budget(20.0, remaining, ALPHA_REMAINING_FRAC);
        let armed = budget(34.0, remaining, ALPHA_REMAINING_FRAC);
        assert!(
            (shipped - 19.2).abs() < 0.1,
            "shipped budget {shipped:.2} should be ~19.2s"
        );
        assert_eq!(
            shipped, armed,
            "raising alpha_secs 20 -> 34 must change NOTHING while the 0.4 fraction \
             binds — this is why the measured spend moved 0.26s and the root bound zero"
        );
    }

    /// Raising the FRACTION is what actually moves the budget.
    #[test]
    fn raising_the_fraction_is_what_frees_the_ascent() {
        let remaining = 48.0;
        let shipped = budget(34.0, remaining, 0.4);
        let raised = budget(34.0, remaining, 0.7);
        assert!(
            raised > shipped,
            "at frac 0.7 the ascent gets {raised:.1}s vs {shipped:.1}s — the freed \
             bi_topk budget can finally reach the root bound"
        );
        // cfg.secs still bounds it: the fraction cannot grant more than asked.
        assert_eq!(budget(20.0, remaining, 0.9), 20.0, "cfg.secs still caps");
    }

    /// Shipped default is the historical literal, so this lands byte-identical.
    #[test]
    fn the_shipped_fraction_is_the_historical_literal() {
        assert!(
            (ALPHA_REMAINING_FRAC - 0.4).abs() < f64::EPSILON,
            "shipped default must remain 0.4 until the trade is measured"
        );
    }
}

#[cfg(test)]
mod channel_alpha_tests {
    use super::{project_step_to_channels, CHANNEL_ALPHA};

    /// The projection must replace every step in a channel with that channel's
    /// MEAN — this is what collapses C*H*W free variables to C.
    #[test]
    fn projection_replaces_each_channel_with_its_mean() {
        // group = 4 (a 2x2 spatial map), two channels, all neurons live.
        let mut steps = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let live: Vec<usize> = (0..8).collect();
        let changed = project_step_to_channels(&mut steps, &live, 4);
        assert_eq!(&steps[0..4], &[2.5, 2.5, 2.5, 2.5], "channel 0 mean is 2.5");
        assert_eq!(
            &steps[4..8],
            &[25.0, 25.0, 25.0, 25.0],
            "channel 1 mean is 25.0"
        );
        assert_eq!(changed, 8, "every step differed from its channel mean");
    }

    /// Only LIVE (unstable) neurons participate — a stable neuron has an exact
    /// fixed line and must not be dragged by, or contribute to, a channel mean.
    #[test]
    fn only_live_neurons_participate() {
        let mut steps = vec![1.0, 2.0, 100.0, 100.0];
        let live = vec![0usize, 1]; // 2 and 3 are stable
        project_step_to_channels(&mut steps, &live, 4);
        assert_eq!(steps[0], 1.5);
        assert_eq!(steps[1], 1.5);
        assert_eq!(steps[2], 100.0, "stable neuron must be untouched");
        assert_eq!(steps[3], 100.0, "stable neuron must be untouched");
    }

    /// A group of 1 (Gemm relu, or a 1x1 spatial map) has no channel structure:
    /// the projection must be a no-op, not a global average.
    #[test]
    fn group_of_one_is_a_no_op() {
        let mut steps = vec![1.0, 2.0, 3.0];
        let live: Vec<usize> = (0..3).collect();
        let changed = project_step_to_channels(&mut steps, &live, 1);
        assert_eq!(changed, 0);
        assert_eq!(
            steps,
            vec![1.0, 2.0, 3.0],
            "group=1 must not average anything"
        );
    }

    /// The projection preserves the SUM of steps within a channel, so it
    /// redistributes the ascent's move rather than inflating or shrinking it.
    #[test]
    fn projection_conserves_the_per_channel_total() {
        let mut steps = vec![0.3, -0.1, 0.8, 0.0];
        let live: Vec<usize> = (0..4).collect();
        let before: f64 = steps.iter().sum();
        project_step_to_channels(&mut steps, &live, 4);
        let after: f64 = steps.iter().sum();
        assert!(
            (before - after).abs() < 1e-12,
            "total step must be conserved"
        );
    }

    /// Ships OFF: the historical per-neuron ascent is untouched.
    #[test]
    fn channel_alpha_ships_off() {
        const { assert!(!CHANNEL_ALPHA, "shipped default must remain per-neuron") };
    }
}

#[cfg(test)]
mod steer_branch_tests {
    use super::{steer_selection, STEER_ACTIVE_BRANCH};

    /// `(d, class, direct_attains_max)`, objective-sorted ascending.
    fn scored() -> Vec<(f64, usize, bool)> {
        vec![
            (-0.90, 7, false), // worst, but PINNED by a frozen term
            (-0.80, 3, false), // also pinned
            (-0.70, 5, true),  // first class alpha can actually move
            (-0.60, 1, true),
            (-0.50, 9, true),
        ]
    }

    #[test]
    fn the_unarmed_selection_is_the_historical_prefix() {
        let (cols, _) = steer_selection(&scored(), 3, false);
        assert_eq!(
            cols,
            vec![7, 3, 5],
            "disarmed must be the plain lowest-d prefix"
        );
    }

    #[test]
    fn arming_drops_the_frozen_pinned_classes_from_the_mean() {
        let (cols, _) = steer_selection(&scored(), 3, true);
        assert_eq!(
            cols,
            vec![5, 1, 9],
            "a class whose objective a frozen term attains has a ZERO supergradient in alpha \
and must not be folded into the mean direction"
        );
    }

    #[test]
    fn the_pinned_census_counts_the_whole_set_not_the_steer_window() {
        // want=1 looks at one column; the census must still report BOTH pinned
        // classes, or the observable would under-report the defect it exists to
        // measure.
        let (cols, pinned) = steer_selection(&scored(), 1, true);
        assert_eq!(cols, vec![5]);
        assert_eq!(pinned, 2, "census is over all classes, not the window");
    }

    #[test]
    fn an_all_pinned_ascent_yields_an_empty_steer_set_and_therefore_stops() {
        let all_pinned: Vec<(f64, usize, bool)> =
            vec![(-0.9, 7, false), (-0.8, 3, false), (-0.7, 5, false)];
        let (cols, pinned) = steer_selection(&all_pinned, 8, true);
        assert!(
            cols.is_empty(),
            "no class can move the objective; `mean_supergradient` returns None on an empty \
column set and the ascent breaks instead of burning the budget"
        );
        assert_eq!(pinned, 3);
        // Unarmed, the same input still proposes iterates — that is the
        // behaviour being replaced, pinned here so the delta is explicit.
        let (cols_off, _) = steer_selection(&all_pinned, 8, false);
        assert_eq!(cols_off, vec![7, 3, 5]);
    }

    #[test]
    fn a_tie_between_direct_and_the_frozen_term_counts_as_active() {
        // `direct == frozen` => the max is attained by BOTH, so d(direct)/d(alpha)
        // is a valid supergradient of `d`. Keeping the class is correct; the
        // eval site spells the comparison `direct[f] >= frozen`.
        let tie: Vec<(f64, usize, bool)> = vec![(-0.5, 2, true)];
        let (cols, pinned) = steer_selection(&tie, 4, true);
        assert_eq!(cols, vec![2]);
        assert_eq!(pinned, 0);
    }

    #[test]
    fn requesting_more_columns_than_exist_is_not_an_error() {
        let (cols, _) = steer_selection(&scored(), 99, true);
        assert_eq!(
            cols,
            vec![5, 1, 9],
            "take() saturates; no panic, no padding"
        );
        let (cols_off, _) = steer_selection(&scored(), 99, false);
        assert_eq!(cols_off, vec![7, 3, 5, 1, 9]);
    }

    #[test]
    fn steer_branch_ships_off() {
        const {
            assert!(
                !STEER_ACTIVE_BRANCH,
                "#steer-branch is an EXPERIMENT constant; the shipped default is false so the \
lane stays arithmetically identical until it is measured"
            )
        };
    }
}
