// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Phase-level time budget configuration for β-CROWN verification.
//!
//! Centralizes the timeout policy that was previously scattered across
//! `attack_budget.rs`, `sequential.rs`, `graph.rs`, `disjunctive.rs`,
//! and `mod.rs` as hardcoded magic constants.
//!
//! Default values are tuned for VNN-COMP competitive performance
//! (20% warmup cap, 10% post-BaB PGD reservation).

use ny_core::{NyError, Result};
use serde::{Deserialize, Serialize};

/// Phase-level time budget fractions for β-CROWN verification.
///
/// Each fraction is relative to the total verification timeout.
/// All callers derive their phase budgets from this single config
/// instead of computing local fractions from raw `timeout`.
///
/// Default values are tuned for VNN-COMP competitive performance:
/// 20% warmup cap, 10% post-BaB PGD reservation.
// `PartialEq` is derived so scoped-preset tests can assert that a preset differs
// from the sealed defaults in EXACTLY the fields it means to change. Comparing the
// whole struct — rather than field by field — keeps those tests honest when a new
// field is added later.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhaseBudgetConfig {
    /// Fraction of total timeout allocated to initial alpha/CROWN/CROWN-IBP bounds.
    ///
    /// Default: `0.20` (warmup gets 20% of BaB timeout).
    /// On a 100s timeout with 10% PGD reservation: warmup gets 18s.
    ///
    /// NOT THE CONTROLLING CAP ON THE MULTI-OBJECTIVE GRAPH ROOT PATH. There,
    /// `multi_objective::root::evaluate_root` deliberately sets
    /// `initial_deadline = Some(bab_deadline)` — the mandatory foundational
    /// node-bounds sweep must reach every node, and capping it at this fraction
    /// choked conv-heavy DAGs (yolo, tinyimagenet) into "deadline exceeded
    /// before node 'Conv_0'" with most of the budget unused. The
    /// `[phase] multiobj-budget` telemetry line prints this fraction beside the
    /// window actually selected precisely so the two are not confused.
    ///
    /// MEASURED consequence (relusplitter/cifar_biasfield, 7cae6103, shipped
    /// preset through `ny vnncomp`, disjunctive attack suppressed so the whole
    /// window was available): row 146 got `effective-bab=118.500s` with this
    /// fraction at 0.200 (nominally a 23.7s warmup) and the bootstrap ran
    /// `graph-bab-bootstrap start t=0.0s` → `end t=118.5s` — 100% of the
    /// window, of which one CROWN-IBP DAG collection was 103.6s and still
    /// reverted 8/9 demanded targets to IBP on per-node deadlines. That is the
    /// design decision above working as written, not a cap-enforcement bug; do
    /// not "fix" it by dividing the measured warmup by `0.20 * total`.
    pub initial_bounds_fraction: f32,

    /// Fraction of total timeout allocated to upfront PGD attack.
    ///
    /// Current behavior: `0.20` (upfront PGD gets 20% of timeout).
    /// Source: `verify/attack_budget.rs:12-17` → `timeout / 5`.
    ///
    /// SCOPE: this caps the UPFRONT PGD phase only — `verify/graph.rs`'s
    /// pre-α-CROWN attack, `verify/sequential.rs`, and the MIP-only warm-start
    /// precheck. It does NOT govern a multi-clause disjunction: those route to
    /// `disjunctive::verify_multi_clause_disjunction`, whose attack phase is
    /// sized by [`Self::disjunctive_pgd_fraction`] (0.50), and `graph.rs` skips
    /// its upfront attack for `is_disjunction && !single_clause` anyway.
    /// Measuring a disjunctive attack phase against THIS fraction reports a
    /// spurious `0.50 / 0.20 = 2.5x` "overrun" — see the `#wrong-cap` guard test
    /// `disjunctive_attack_cap_is_not_the_upfront_cap` in
    /// `ny-cli/src/commands/beta_crown/verify/phase_budget.rs`.
    pub upfront_pgd_fraction: f32,

    /// Fraction of total timeout allocated to reduced verification
    /// (sequential path only).
    ///
    /// Current behavior: `0.40` (sequential reduced verification gets 40%).
    /// Source: `verify/sequential.rs:89-113` → `timeout * 2 / 5`.
    pub reduced_verification_fraction: f32,

    /// Fraction of total timeout allocated to disjunctive global PGD.
    ///
    /// Current behavior: `0.50`.
    /// Source: `verify/disjunctive.rs:96-126`.
    pub disjunctive_pgd_fraction: f32,

    /// Fraction of total timeout allocated to disjunctive CROWN/alpha precheck.
    ///
    /// Current behavior: `0.20`.
    /// Source: `verify/disjunctive.rs:197-289`.
    pub disjunctive_precheck_fraction: f32,

    /// Minimum fraction of total timeout guaranteed for MIP fallback.
    ///
    /// Current behavior: `0.25` → `timeout / 4`.
    /// Source: `beta_crown/mod.rs:1193-1242`.
    pub mip_min_fraction: f32,

    /// Minimum MIP timeout in seconds (floor clamp).
    ///
    /// Current behavior: `5`.
    /// Source: `beta_crown/mod.rs:1193-1242`.
    pub mip_min_secs: u64,

    /// Maximum MIP timeout in seconds (ceiling clamp).
    ///
    /// Current behavior: `30`.
    /// Source: `beta_crown/mod.rs:1193-1242`.
    pub mip_max_secs: u64,

    /// Fraction of MIP timeout allocated to CROWN-IBP preprocessing.
    ///
    /// Current behavior: `0.05`.
    /// Source: `mip_highs_intermediate_bounds.rs:6-33`.
    pub mip_crown_ibp_fraction: f64,

    /// Minimum MIP CROWN-IBP preprocessing timeout in seconds.
    ///
    /// Current behavior: `0.25`.
    /// Source: `mip_highs_intermediate_bounds.rs:6-33`.
    pub mip_crown_ibp_min_secs: f64,

    /// Maximum MIP CROWN-IBP preprocessing timeout in seconds.
    ///
    /// Current behavior: `2.0`.
    /// Source: `mip_highs_intermediate_bounds.rs:6-33`.
    pub mip_crown_ibp_max_secs: f64,

    /// Minimum per-clause timeout in seconds.
    ///
    /// Current behavior: `1` (implicit floor in per-constraint iteration).
    pub min_clause_secs: u64,

    /// Fraction of total timeout reserved for post-BaB PGD attack.
    ///
    /// When BaB returns Timeout or Unknown, the engine tries PGD as a
    /// last-resort counterexample search. Without a reservation, BaB
    /// consumes the entire timeout and PGD's deadline is already expired.
    ///
    /// The BaB loop stops at `timeout * (1 - post_bab_pgd_fraction)` so
    /// PGD gets the remaining `timeout * post_bab_pgd_fraction`.
    ///
    /// Default: `0.10` (reserve 10% for PGD fallback).
    /// On a 100s timeout: BaB gets 90s, post-BaB PGD gets 10s.
    ///
    /// #reserve-demand: this fraction is a REQUEST, not an entitlement. The
    /// caller ledger drops it when a purely static scan proves the lane that
    /// would spend it cannot take a step on this net — see
    /// [`Self::demand_sized_post_bab_reserve`].
    pub post_bab_pgd_fraction: f32,

    /// #reserve-demand — size the post-BaB reservation from the lane that
    /// actually follows BaB, instead of always subtracting the constant
    /// [`Self::post_bab_pgd_fraction`].
    ///
    /// A reserve carved out BEFORE BaB can only be justified by a capability
    /// that is known BEFORE BaB. The post-BaB falsification lane needs either a
    /// network gradient (statically decidable: the exact-gradient fragment) or a
    /// seed the verifier has not produced yet. When the gradient is statically
    /// absent, the pre-payment buys nothing: the lane is handed the window and
    /// declines in ~0.03 s (`#deadlane`), and the seconds are simply discarded
    /// with the verdict still undecided. Measured on vit_2023 at the official
    /// 100 s budget: BaB stopped at 81.0 s of a 95 s internal tier, the 14 s
    /// tail was refused by both consumers, and the row published `timeout`
    /// while its own ledger still read `remaining_s=14.198`.
    ///
    /// This never grants time: it only declines to TAKE time from BaB, and the
    /// BaB deadline still moves no later than `overall_deadline`. A lane that
    /// can run is unaffected — the seed-driven stages still receive whatever
    /// wall remains after BaB, they are simply not pre-paid.
    ///
    /// `false` restores the historical unconditional subtraction exactly.
    pub demand_sized_post_bab_reserve: bool,

    /// #seq-single-reserve — base the SEQUENTIAL engine's warmup cap on the
    /// caller's ledger-true BaB boundary instead of double-reserving the
    /// post-BaB PGD slice.
    ///
    /// When the caller passes an explicit wall-clock deadline, that
    /// deadline is the caller ledger's ALREADY-RESERVED BaB boundary — the
    /// ledger has already subtracted [`Self::post_bab_pgd_fraction`] and the
    /// MIP reserve (`phase_budget.rs::bab_deadline`). Every graph lane
    /// refuses to reserve again ("do not reserve post-BaB time a second
    /// time" — `relu_split/bab_loop.rs`, and the same treatment on the
    /// input-split and gpu_bab lanes). The sequential initial phase
    /// (`verify_phases/initial.rs`) nevertheless derived its warmup cap from
    /// `effective_total * (1 - post_bab_pgd_fraction) * initial_bounds_fraction`
    /// — the PGD reserve taken a SECOND time.
    ///
    /// MEASURED 2026-08-31 (two-point validation to 0.1 s, `reports/
    /// measured-2026/tll_n64_attack_starvation_2026-08-31/README.md`): on a
    /// tll 600 s row the ledger hands the engine 483.0 s, the double reserve
    /// cuts it to 434.7 s, and the 0.85 warmup fraction lands the cap at
    /// 369.5 s + ~2.8 s setup = 372.3 s predicted vs 372.38/372.36 measured
    /// (and 437.4 predicted vs 437.44/437.46 at fraction 1.00).
    ///
    /// SCOPE — this key corrects the WARMUP CAP ONLY. The BaB search
    /// boundary (`bab_timeout`) deliberately KEEPS the engine's own single
    /// reserve, because on the sequential route the window
    /// `[bab_timeout, caller deadline]` is the engine-internal post-BaB PGD
    /// fallback's ONLY home ("this deadline extends past the BaB timeout so
    /// PGD has guaranteed time" — `engine/pgd.rs`), and it is the lane that
    /// produced tll `N=56_3`'s banked sat at +481 s of a 484.9 s run. Moving
    /// the search boundary onto the caller deadline would zero that window
    /// and forfeit exactly the attack #seq-early-attack exists to arm.
    ///
    /// The warmup cap under this key only ever moves LATER (the historical
    /// base is `effective_total * (1 - f)`, the corrected base is
    /// `effective_total`) and never past the caller deadline. Scheduling
    /// only: no bound or verdict math is touched.
    ///
    /// `false` restores the historical double-reserved warmup cap
    /// byte-identically.
    pub sequential_warmup_single_reserve: bool,

    /// #seq-early-attack — route a SEQUENTIAL initial-phase deadline expiry
    /// through the engine's post-BaB PGD fallback instead of forfeiting it.
    ///
    /// The initial phase translates a warmup-cap expiry
    /// (`NyError::DeadlineExceeded`) into an `Early(Timeout)` outcome, and
    /// `engine/core.rs` returns that outcome BEFORE the `pgd_fallback`
    /// closure is constructed — so an early-quit row never runs a single PGD
    /// step, while the in-phase timeout path a few lines below feeds the
    /// SAME result through `try_pgd_attack_with_deadline`. Measured cost
    /// (tll, 2026-08-31): the two field-sat N=64 rows spend 100% of a 600 s
    /// budget on a bounds phase that provably cannot finish and 0 s
    /// attacking, while the attack that produced the banked `N=56_3` sat
    /// cost 2.24 s.
    ///
    /// Containment is structural: the fallback can only ever produce a
    /// `Violated` carrying an independently re-validated counterexample
    /// (`try_pgd_attack_with_deadline` re-runs a clean concrete forward and
    /// returns the ORIGINAL result otherwise), and the CLI still gates any
    /// sat through its trusted oracle. It cannot break a proof; its only
    /// cost is wall time the early quit was already discarding.
    ///
    /// `false` restores the historical forfeit byte-identically.
    pub sequential_early_quit_attack: bool,

    /// #mip-demand — size the Graph-MIP escalation reservation from the
    /// escalation's own STATIC admission test, instead of always subtracting
    /// [`Self::mip_min_fraction`]'s slice.
    ///
    /// Sibling of [`Self::demand_sized_post_bab_reserve`], and the same
    /// argument. `bab_deadline` carves the MIP slice out of the SCORED budget
    /// before BaB runs, so it can only be justified by a capability known
    /// before BaB. Two of the escalation gate's own predicates are exactly
    /// that:
    ///
    /// 1. the whole-net encode-nnz cap (`NY_GRAPH_MIP_MAX_NNZ`, 5M) — `nnz` is
    ///    a property of the MODEL, identical on every row of a family;
    /// 2. the minimum-slice floor (`NY_GRAPH_MIP_MIN_SLICE_S`, 20 s) — a
    ///    reserve smaller than the floor is refused by the escalation's first
    ///    gate whatever the model is.
    ///
    /// MEASURED 2026-08-30 (`reports/measured-2026/stop_census_2026-08-30/`,
    /// 7/7 rows produced the `[stop]` line): `challenging_certified` idx1031
    /// reserved 40 s of a 114 s solver deadline and the escalation declined in
    /// ~2 ms with `estimated nnz 150936750 > cap 5000000`; `metaroom_2023`
    /// idx125 reserved 54 s of 200 s and declined with `nnz 50355927`;
    /// `cora_2024` img141 reserved 6 s of 25 s and declined at the 20 s slice
    /// floor before the model was ever looked at.
    ///
    /// RE-DERIVED, because those `mip_timeout=` figures are TWO reserves: the
    /// Graph-MIP slice and the separate `post_bab_pgd_fraction` slice. The part
    /// THIS key governs is the MIP slice alone — 28 s of challenging's 114 s
    /// solver deadline, 30 s (clamped) of metaroom's 200 s, 6 s of cora's 25 s.
    /// That is 15–25% of the solver deadline returned, i.e. **+20% to +37% more
    /// BaB search**, not the census's 24–35%.
    ///
    /// This never grants time: it only declines to TAKE time from BaB, and the
    /// BaB deadline still moves no later than `overall_deadline`. A model the
    /// escalation could actually encode is unaffected — the static probe, not
    /// the yaml, is what drops the reserve, and it is a LOWER bound on the
    /// runtime estimate so it can only fire where the runtime gate would.
    ///
    /// `false` restores the historical unconditional reservation exactly.
    pub demand_sized_mip_reserve: bool,

    /// #disj-demand — zero the disjunctive global PGD slice when the net is
    /// STATICALLY outside the exact-gradient fragment, i.e. when the same
    /// static probe that drives [`Self::demand_sized_post_bab_reserve`]
    /// (`GraphNetwork::supports_attack_point_gradient`) says no exact-gradient
    /// attack step can ever run on this net.
    ///
    /// On such a net the disjunctive attack phase cannot use its exact-VJP
    /// lanes; it degrades to SPSA over point-IBP forward evaluations. MEASURED
    /// on vit_2023 (both `ibp_3_3_8` and `pgd_2_3_16` are outside the
    /// fragment) at the official 100 s budget: the phase consumes its full
    /// `disjunctive_pgd_fraction` slice (4.75 s of a 95 s tier, ~45 ms per
    /// point-IBP eval ⇒ ~35 SPSA steps) and has NEVER produced a
    /// counterexample on the family — zero `sat` in every campaign run, and
    /// the 2026-08-25 independent adjudication searched ~1.9 M evaluations per
    /// row without finding a violation. Zeroing the slice hands those seconds
    /// to BaB, which re-bases on `ledger.remaining()`.
    ///
    /// Default `false` (historical schedule, byte-identical everywhere): the
    /// SPSA fallback is a REAL searcher and other gradient-ineligible
    /// categories (ml4acopf_2024 has a banked 256 s `sat`) may depend on it,
    /// so a category must arm this through its own measured preset. On a net
    /// INSIDE the fragment the flag is inert even when armed — the static
    /// probe, not the yaml, is what zeroes the slice.
    pub demand_sized_disjunctive_pgd: bool,

    /// Fraction of the REMAINING budget granted to the single adaptive
    /// attack-phase extension (#attack-extend) when the disjunctive attack was
    /// deadline-cut with a promising margin.
    ///
    /// Default: `0.15` (batteries-included ON). Set `0.0` in a preset to
    /// disable the extension for categories where the margin gate cannot
    /// discriminate — e.g. cgan_2023 band properties, where the reachable set
    /// hugs the disjunctive thresholds so every UNSAT instance reports a
    /// "promising" margin and the extension only starves the bound phases.
    /// `NY_ATTACK_EXTEND_FRAC` still overrides this at runtime;
    /// `NY_ATTACK_EXTEND=0` remains the global kill switch.
    pub attack_extension_fraction: f32,
    /// Optional ABSOLUTE ceiling (seconds) on the disjunctive global PGD phase,
    /// applied on top of [`Self::disjunctive_pgd_fraction`].
    ///
    /// The disjunctive PGD deadline is `start + total*frac`, anchored at the
    /// ledger start. On a HARD UNSAT (hold) instance the falsifier never finds a
    /// counterexample and runs its whole slice, so at large budgets `frac*total`
    /// balloons — e.g. cifar100 at a 300s budget hands PGD 0.40*300 = 120s of
    /// dead falsification before BaB, and a concurrent forward-linear warmer
    /// under `thread::scope` can stall the phase across that window. Since the
    /// disjunctive fast-path BaB re-bases on `ledger.remaining()`, any second
    /// not spent in PGD is automatically reclaimed for BaB.
    ///
    /// `Some(secs)` caps the phase at `min(frac*total, secs)` — a robustness
    /// floor for hold-heavy conv benchmarks where PGD beyond a few seconds is
    /// wasted (sat instances are falsified in a few seconds). `None` (default)
    /// preserves the pure-fraction behavior for every other benchmark — no
    /// regression. Only the disjunctive global PGD reads this; upfront PGD is
    /// untouched.
    #[serde(default)]
    pub disjunctive_pgd_max_secs: Option<u64>,

    /// Optional ABSOLUTE floor (seconds) on the disjunctive global PGD phase,
    /// applied AFTER the tiny-budget attack cap
    /// (`verify::phase_budget::SMALL_BUDGET_ATTACK_FRACTION`).
    ///
    /// The tiny-budget cap holds every attack phase to 15 % of a `<= 30 s`
    /// budget so a falsifier cannot starve the bound phases. On a category where
    /// BaB can never decide, that cap is pure loss: MEASURED on lsnc_relu
    /// (20 s internal budget → 3.0 s attack slice, 80 rows, ZERO unsat ever
    /// produced), the disjunctive PGD lands `quadrotor2d_state_34`'s
    /// ORT-confirmed counterexample ~4.54 s into the phase, i.e. 1.5 s past the
    /// cap — and BaB then spends the remaining 15 s doubling its input-split
    /// queue to 30 762 domains before dying inside a batch on the wall deadline,
    /// returning NO verdict at all.
    ///
    /// `Some(secs)` raises the phase to `max(capped, min(secs, total/2))`;
    /// BaB therefore always keeps at least half the scored budget, and the
    /// floor can never pass the overall deadline. `None` (default) is
    /// byte-identical to before this knob existed. Only the disjunctive global
    /// PGD reads it — upfront PGD keeps the plain cap.
    #[serde(default)]
    pub disjunctive_pgd_min_secs: Option<u64>,

    /// #attack-anchor: measure the disjunctive global PGD slice from the PHASE
    /// START instead of the LEDGER START.
    ///
    /// [`Self::disjunctive_pgd_fraction`] is normally spent as
    /// `start + total*frac`, an ABSOLUTE instant fixed when the instance
    /// begins. Everything that happens before the falsification phase — model
    /// load, graph build, VNN-LIB parse, spec lowering — is therefore charged
    /// against the attack's own slice, and on a big model it can consume all of
    /// it. MEASURED on `cifar100_2024` `CIFAR100_resnet_large` at the official
    /// 100 s budget with the shipped `disjunctive_pgd_fraction: 0.05`
    /// (`[pgd-vjp-disj]` diagnostic):
    ///
    /// ```text
    /// deadline (0.1s): wave_steps=0 avg fwd=0.0ms vjp=0.0ms best_margin=-0.15517
    /// ```
    ///
    /// The batched exact-VJP falsifier received **0.1 s of a 5 s slice and took
    /// ZERO steps** — the 15 MB ONNX had already burned ~4.9 s. That is why the
    /// preset's declared 5 % falsification budget does not exist in practice on
    /// the large model while the smaller `resnet_medium` (whose setup is
    /// cheaper) still falsifies inside the same slice.
    ///
    /// This is the same defect the CROWN precheck already fixed with
    /// `PhaseBudgetLedger::phase_deadline_from_now` (#four-walls: `start +
    /// total*0.20` "was already in the past after a 50% attack phase"), one
    /// phase over.
    ///
    /// `true` grants `now + total*frac` (still clamped by the tiny-budget
    /// attack cap, by [`Self::disjunctive_pgd_max_secs`] measured from the
    /// same phase start, and by the overall deadline). It gives the phase what
    /// the preset ASKED for — never more.
    ///
    /// Completeness-only by construction: this changes how long the falsifier
    /// runs and nothing else. An attack can only ever propose a counterexample
    /// CANDIDATE, every candidate still passes `re_evaluate_and_confirm` and
    /// the trusted-ORT vnncomp gate, and no `unsat` is ever concluded from an
    /// attack. It cannot narrow a bound or mint a verdict.
    ///
    /// `false` (default) is byte-identical to before this knob existed.
    #[serde(default)]
    pub disjunctive_pgd_from_phase_start: bool,

    /// Optional ABSOLUTE ceiling (seconds) on the disjunctive CROWN/alpha
    /// PRECHECK phase, applied on top of
    /// [`Self::disjunctive_precheck_fraction`] (#precheck-abs-cap).
    ///
    /// The precheck slice is `now + total*frac` (from NOW, not the ledger
    /// start — see `PhaseBudgetLedger::phase_deadline_from_now`), so the
    /// fraction bounds NOTHING relative to what is actually left: with the
    /// default policy the disjunctive PGD phase already ends at `0.50*total`
    /// and the precheck is then handed a further `0.20*total` from there,
    /// followed by an equally-sized alpha-precheck slice — 0.90 of the budget
    /// can legally be spent before BaB starts. Worse, the phase grows with the
    /// MODEL (the cgan_2023 CROWN-IBP collection went from ~150 s to ~5x that
    /// as the collector got wider targets), so a fraction tuned when the work
    /// was cheap becomes a pure BaB-starvation lever once the work is not.
    ///
    /// `Some(secs)` caps the phase at `min(frac*total, secs)` measured from
    /// the phase start — the precheck keeps a slice big enough for the work it
    /// is known to need, and every second it does not spend is reclaimed by
    /// BaB (which re-bases on `ledger.remaining()`). `None` (default)
    /// preserves the pure-fraction behavior for every other benchmark — no
    /// regression. Read only by the disjunctive precheck phases.
    #[serde(default)]
    pub disjunctive_precheck_max_secs: Option<u64>,

    /// Optional ADAPTIVE stall cutoff for the disjunctive global PGD phase,
    /// as a fraction of that phase's OWN slice (#attack-stall, design S4).
    ///
    /// `Some(w)`: while the attack runs, watch the best (closest-to-violation)
    /// clause margin — a running MAX over every evaluated point, so it can only
    /// ever RISE. If a whole window of `w * attack_slice` passes without that
    /// maximum rising by more than the confirmation path's own noise floor
    /// (`verify::disjunctive_pgd::noise_scaled_margin`, the measured ny-vs-ORT
    /// forward disagreement), the ascent has plateaued and the attack stops.
    /// Every second it does not spend is reclaimed automatically, because the
    /// fast-path BaB re-bases on `PhaseBudgetLedger::remaining()`. `None`
    /// (default) is byte-identical to before this knob existed.
    ///
    /// This is the per-INSTANCE counterpart of the per-benchmark
    /// [`Self::disjunctive_pgd_max_secs`]: the right split between falsifying
    /// and bounding is a property of the row, not of the category.
    ///
    /// WHY A RATE AND NOT A LEVEL. The margin LEVEL cannot separate the two
    /// populations — measured on metaroom_2023: `spec_idx_28` (GT=**unsat**)
    /// plateaus at **-0.033**, its supremum, while `spec_idx_129` (GT=**sat**)
    /// was still ascending at **-0.083** when its preset cap cut it. The unsat
    /// row is CLOSER to violation than the sat row, so any level threshold gets
    /// one population wrong. The improvement RATE orders that same pair
    /// correctly: flat across 14 restarts versus climbing when cut.
    ///
    /// WHY IT IS UNSET EVERYWHERE. The decisive A/B for the budget it reclaims
    /// came back NEGATIVE — 3 GT-unsat oval21 rows, attack slice 0.50 vs a
    /// capped 0.10/5 s, `unknown 57 s` in BOTH arms on all three
    /// (`docs/CONV_CROWN_WALL_DESIGN_2026-07-27.md` S4; `relusplitter.yaml`
    /// repeats it: "It converted ZERO rows"). And the opposite error is
    /// expensive: on tinyimagenet this same lane is what FINDS the sat rows,
    /// which land at 12.35 s / 17.67 s / 20.39 s of a 100 s budget (b61b5f10,
    /// 8/15 -> 3/15 when a sibling's allocation was ported in). Arm it for a
    /// category only behind an A/B for THAT category that covers its sat rows.
    ///
    /// COMPLETENESS-ONLY. This bounds an ATTACK phase. An attack only ever
    /// produces a counterexample candidate — every `sat` still passes
    /// re-evaluation and the trusted-ORT gate — and no `unsat` is ever
    /// concluded from an attack that failed or was cut. Cutting one short can
    /// lose a `sat` (completeness); it can never narrow a bound or mint a
    /// verdict.
    #[serde(default)]
    pub disjunctive_pgd_stall_window_fraction: Option<f32>,

    /// #mip-handoff — ENFORCE the MIP reservation this policy already declares.
    ///
    /// [`Self::mip_min_fraction`] / [`Self::mip_min_secs`] size a slice that
    /// `PhaseBudgetLedger::bab_deadline` carves out of the scored budget for the
    /// exact-MIP complete verifier. That carve-out is only a *plan*: the
    /// same-LHS relational reduction (`verify::sequential::try_reduced_verification`)
    /// historically receives NO absolute deadline and owns the whole remaining
    /// wall clock, because ACAS-Xu prop_2 needs it. On a category whose closer
    /// is MIP rather than BaB, that means the declared reservation is never
    /// handed over: BaB runs to the internal deadline and the escalation gate
    /// (`dispatch.rs`, `mip_timeout >= 5`) sees zero seconds.
    ///
    /// MEASURED 2026-08-06, safenlp_2024 `ruarobot/hyperrectangle_3558`,
    /// official 20 s budget (internal tier 15 s), `NY_PHASE_TELEMETRY=1`:
    ///   `bab_end_s=7.500 ... mip_reserved_s=9.000` planned, BaB actually
    ///   returned at `elapsed_s=15.312`, then
    ///   `MIP escalation gate: ... mip_timeout=0s remaining=0s` — the preset's
    ///   entire declared strategy (`complete_verifier=mip` equivalent) never ran
    ///   on ANY row of the category.
    ///
    /// `true` makes the same-LHS BaB stop at the reserved handoff so the
    /// escalation actually starts, and drops the two allocations that would
    /// otherwise shrink the grant back under the gate (the half-budget BaB floor
    /// and the 3 s post-BaB attack tail — the latter is redundant for categories
    /// that win their `sat` rows in an UPFRONT attack lane).
    ///
    /// BUDGET-ONLY, therefore VERDICT-NEUTRAL: it moves a deadline INSIDE the
    /// same scored budget and never past `overall_deadline`. Reallocating time
    /// cannot make a bound wrong; BaB exiting early is `Unknown`, which is sound,
    /// and MIP is an exact complete verifier. `false` (default) is byte-identical
    /// to before this knob existed for every other category.
    #[serde(default)]
    pub enforce_mip_handoff: bool,
}

impl PhaseBudgetConfig {
    /// Validate every value that participates in duration arithmetic.
    ///
    /// Duration multiplication requires a finite, non-negative factor. Most
    /// consumers historically clamped their local copy, which both hid
    /// malformed presets and left NaN able to reach `Duration::mul_f32`.
    /// Reject once at the shared config boundary so every engine ingress has
    /// the same total policy and no malformed schedule can panic a scored run.
    pub fn validate(&self) -> Result<()> {
        fn finite_f32(name: &str, value: f32) -> Result<()> {
            if !value.is_finite() {
                return Err(NyError::InvalidConfig(format!(
                    "phase_budget.{name} must be finite, got {value}"
                )));
            }
            Ok(())
        }

        fn nonnegative_f64(name: &str, value: f64) -> Result<()> {
            if !value.is_finite() || value < 0.0 {
                return Err(NyError::InvalidConfig(format!(
                    "phase_budget.{name} must be finite and >= 0, got {value}"
                )));
            }
            Ok(())
        }

        fn finite_f64(name: &str, value: f64) -> Result<()> {
            if !value.is_finite() {
                return Err(NyError::InvalidConfig(format!(
                    "phase_budget.{name} must be finite, got {value}"
                )));
            }
            Ok(())
        }

        // These fields have long-standing local clamp semantics. Preserve
        // finite out-of-range compatibility, but reject NaN/infinity before it
        // can survive a clamp and reach duration arithmetic.
        finite_f32("initial_bounds_fraction", self.initial_bounds_fraction)?;
        finite_f32("upfront_pgd_fraction", self.upfront_pgd_fraction)?;
        finite_f32(
            "reduced_verification_fraction",
            self.reduced_verification_fraction,
        )?;
        finite_f32("disjunctive_pgd_fraction", self.disjunctive_pgd_fraction)?;
        finite_f32(
            "disjunctive_precheck_fraction",
            self.disjunctive_precheck_fraction,
        )?;
        finite_f32("mip_min_fraction", self.mip_min_fraction)?;
        // Both engine consumers have always clamped this reservation to
        // [0, 0.5]. Preserve that finite compatibility; the outer wrapper
        // resolves proof-tail ownership from the same effective clamped value.
        finite_f32("post_bab_pgd_fraction", self.post_bab_pgd_fraction)?;
        finite_f32("attack_extension_fraction", self.attack_extension_fraction)?;

        // An inverted whole-net MIP range is an existing typed DISARM signal
        // (`requests_mip_reservation` returns false), so it remains valid.
        finite_f64("mip_crown_ibp_fraction", self.mip_crown_ibp_fraction)?;
        nonnegative_f64("mip_crown_ibp_min_secs", self.mip_crown_ibp_min_secs)?;
        nonnegative_f64("mip_crown_ibp_max_secs", self.mip_crown_ibp_max_secs)?;
        if self.mip_crown_ibp_min_secs > self.mip_crown_ibp_max_secs {
            return Err(NyError::InvalidConfig(format!(
                "phase_budget.mip_crown_ibp_min_secs ({}) must be <= \
                 mip_crown_ibp_max_secs ({})",
                self.mip_crown_ibp_min_secs, self.mip_crown_ibp_max_secs
            )));
        }
        Ok(())
    }

    /// Whether this policy requests a nonzero whole-network MIP reservation.
    ///
    /// This is the shared admission predicate for the pre-BaB time slice, the
    /// root-bounds reuse stash, and whole-net Graph-MIP in Auto mode. A zero
    /// reservation must not trigger a potentially large bounds-map clone or a
    /// late bounds recompute merely because Graph-MIP is enabled; explicit MIP
    /// remains an override. The independent per-leaf MIP oracle receives its
    /// bounds directly and is deliberately outside this policy.
    pub fn requests_mip_reservation(&self) -> bool {
        self.mip_max_secs > 0
            && self.mip_min_secs <= self.mip_max_secs
            && ((self.mip_min_fraction.is_finite() && self.mip_min_fraction > 0.0)
                || self.mip_min_secs > 0)
    }
}

/// The sequential initial phase's derived time schedule (#seq-single-reserve).
///
/// Produced by [`PhaseBudgetConfig::sequential_initial_schedule`] so the
/// arithmetic the engine runs is the same arithmetic the tests pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequentialInitialSchedule {
    /// BaB search boundary measured from the phase start. Unchanged by
    /// #seq-single-reserve in BOTH dispositions: this is the engine's own
    /// single post-BaB PGD reserve, the internal fallback's only window.
    pub bab_timeout: std::time::Duration,
    /// Warmup (initial-bounds) cap measured from the phase start.
    pub warmup_cap: std::time::Duration,
    /// Whether the warmup cap was based on the caller's ledger-true boundary
    /// (`effective_total`) rather than the double-reserved `bab_timeout`.
    pub warmup_single_reserve_applied: bool,
}

impl PhaseBudgetConfig {
    /// Derive the sequential initial phase's schedule from the effective
    /// total window (#seq-single-reserve).
    ///
    /// `caller_deadline` is whether the engine was handed an explicit
    /// wall-clock deadline — i.e. whether `effective_total` is a
    /// caller ledger's already-reserved BaB boundary. Only then is there a
    /// second reserve to decline; a config-timeout run keeps the engine's
    /// own reserve as the single one, byte-identically.
    ///
    /// Invariants, asserted by test on a grid including the measured tll
    /// two-point validation:
    /// * `bab_timeout` is the historical `effective_total * (1 - pgd_frac)`
    ///   in EVERY disposition;
    /// * `warmup_cap` never moves EARLIER than the historical cap and never
    ///   past `effective_total`;
    /// * with the kill switch off, or without a caller deadline, both fields
    ///   reproduce the historical formulas byte-identically.
    pub fn sequential_initial_schedule(
        &self,
        effective_total: std::time::Duration,
        caller_deadline: bool,
    ) -> SequentialInitialSchedule {
        let pgd_frac = self.post_bab_pgd_fraction.clamp(0.0, 0.5);
        let bab_timeout = effective_total.mul_f32(1.0 - pgd_frac);
        let warmup_single_reserve_applied =
            caller_deadline && self.sequential_warmup_single_reserve;
        let warmup_base = if warmup_single_reserve_applied {
            // The caller's deadline already sits behind its ledger's post-BaB
            // reserve; do not reserve post-BaB time a second time (the graph
            // lanes' treatment, applied to the warmup cap).
            effective_total
        } else {
            bab_timeout
        };
        let frac = self.initial_bounds_fraction.clamp(0.0, 1.0);
        SequentialInitialSchedule {
            bab_timeout,
            warmup_cap: warmup_base.mul_f32(frac),
            warmup_single_reserve_applied,
        }
    }
}

impl Default for PhaseBudgetConfig {
    fn default() -> Self {
        Self {
            initial_bounds_fraction: 0.20,
            upfront_pgd_fraction: 0.20,
            reduced_verification_fraction: 0.40,
            disjunctive_pgd_fraction: 0.50,
            disjunctive_precheck_fraction: 0.20,
            mip_min_fraction: 0.25,
            mip_min_secs: 5,
            mip_max_secs: 30,
            mip_crown_ibp_fraction: 0.05,
            mip_crown_ibp_min_secs: 0.25,
            mip_crown_ibp_max_secs: 2.0,
            min_clause_secs: 1,
            post_bab_pgd_fraction: 0.10,
            // #reserve-demand: default ON. Byte-identical on every net inside
            // the exact-gradient fragment; only a net whose post-BaB lane is
            // statically dead keeps the seconds in BaB.
            demand_sized_post_bab_reserve: true,
            // #seq-single-reserve: default ON. Warmup-cap-only; the BaB search
            // boundary and the engine PGD window are byte-identical either way.
            sequential_warmup_single_reserve: true,
            // #seq-early-attack: default ON. Adds an attack to a window that
            // was previously discarded; a row the attack cannot falsify
            // returns the identical Timeout.
            sequential_early_quit_attack: true,
            // #mip-demand: default ON. Byte-identical on every model the
            // Graph-MIP encoder could admit; only a model the escalation's own
            // static predicates already refuse keeps the seconds in BaB.
            demand_sized_mip_reserve: true,
            // #disj-demand: default OFF — the SPSA fallback the zeroing removes
            // is a real searcher (ml4acopf's banked 256 s sat is on a
            // gradient-ineligible net). Categories arm it per measured preset.
            demand_sized_disjunctive_pgd: false,
            attack_extension_fraction: 0.15,
            disjunctive_pgd_max_secs: None,
            disjunctive_pgd_min_secs: None,
            // #attack-anchor: the historical LEDGER-START anchoring. `true`
            // requires a per-category A/B — see the field docs.
            disjunctive_pgd_from_phase_start: false,

            disjunctive_precheck_max_secs: None,
            // #attack-stall: the adaptive cutoff ships INERT. Its decisive
            // oval21 A/B converted nothing and the opposite error costs
            // tinyimagenet sat rows — see the field docs.
            disjunctive_pgd_stall_window_fraction: None,
            // #mip-handoff: OFF by default. Same-LHS reduction keeps its full
            // historical wall clock everywhere except categories that opt in.
            enforce_mip_handoff: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_rejects_nonfinite_fractions() {
        macro_rules! assert_invalid {
            ($field:ident, $value:expr) => {{
                let mut cfg = PhaseBudgetConfig::default();
                cfg.$field = $value;
                let error = cfg.validate().expect_err(concat!(
                    "invalid ",
                    stringify!($field),
                    " must be rejected"
                ));
                assert!(
                    error.to_string().contains(stringify!($field)),
                    "field-specific error expected, got {error}"
                );
            }};
        }

        assert_invalid!(initial_bounds_fraction, f32::NAN);
        assert_invalid!(upfront_pgd_fraction, f32::INFINITY);
        assert_invalid!(reduced_verification_fraction, f32::NAN);
        assert_invalid!(disjunctive_pgd_fraction, f32::INFINITY);
        assert_invalid!(disjunctive_precheck_fraction, f32::NEG_INFINITY);
        assert_invalid!(mip_min_fraction, f32::NAN);
        assert_invalid!(post_bab_pgd_fraction, f32::NAN);
        assert_invalid!(post_bab_pgd_fraction, f32::NEG_INFINITY);
        assert_invalid!(attack_extension_fraction, f32::INFINITY);
    }

    #[test]
    fn validation_rejects_inverted_or_nonfinite_mip_ranges() {
        let mut cfg = PhaseBudgetConfig {
            mip_crown_ibp_fraction: f64::NAN,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        cfg = PhaseBudgetConfig {
            mip_crown_ibp_min_secs: 2.1,
            mip_crown_ibp_max_secs: 2.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        cfg = PhaseBudgetConfig {
            mip_crown_ibp_min_secs: f64::NEG_INFINITY,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        PhaseBudgetConfig {
            mip_min_secs: 31,
            mip_max_secs: 30,
            ..Default::default()
        }
        .validate()
        .expect("inverted whole-net MIP range is the established typed disarm signal");
    }

    #[test]
    fn validation_preserves_finite_local_clamp_compatibility() {
        PhaseBudgetConfig {
            initial_bounds_fraction: -1.0,
            upfront_pgd_fraction: 2.0,
            reduced_verification_fraction: -1.0,
            disjunctive_pgd_fraction: 2.0,
            disjunctive_precheck_fraction: -1.0,
            mip_min_fraction: 2.0,
            post_bab_pgd_fraction: -0.01,
            attack_extension_fraction: 1.0,
            ..Default::default()
        }
        .validate()
        .expect("finite legacy values retain their documented local clamp semantics");

        PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.51,
            ..Default::default()
        }
        .validate()
        .expect("the engine retains its historical half-budget ceiling clamp");
    }

    #[test]
    fn validation_accepts_documented_fraction_boundaries() {
        let cfg = PhaseBudgetConfig {
            initial_bounds_fraction: 1.0,
            upfront_pgd_fraction: 0.0,
            reduced_verification_fraction: 1.0,
            disjunctive_pgd_fraction: 0.0,
            disjunctive_precheck_fraction: 1.0,
            mip_min_fraction: 0.0,
            post_bab_pgd_fraction: 0.5,
            attack_extension_fraction: 0.5,
            mip_min_secs: 0,
            mip_max_secs: 0,
            mip_crown_ibp_fraction: 1.0,
            mip_crown_ibp_min_secs: 0.0,
            mip_crown_ibp_max_secs: 0.0,
            ..Default::default()
        };
        cfg.validate().expect("documented boundaries are valid");
    }

    #[test]
    fn default_config_matches_current_behavior() {
        let cfg = PhaseBudgetConfig::default();
        assert!(
            cfg.requests_mip_reservation(),
            "the default/MSCN policy must retain its whole-net MIP reservation"
        );
        // attack_budget.rs: timeout / 5 = 0.20
        assert!((cfg.upfront_pgd_fraction - 0.20).abs() < 1e-6);
        // sequential.rs: timeout * 2 / 5 = 0.40
        assert!((cfg.reduced_verification_fraction - 0.40).abs() < 1e-6);
        // disjunctive.rs: 50% for PGD
        assert!((cfg.disjunctive_pgd_fraction - 0.50).abs() < 1e-6);
        // #disj-demand: historical schedule by default — the SPSA fallback is a
        // real searcher, so demand-zeroing the disjunctive slice is per-preset.
        assert!(!cfg.demand_sized_disjunctive_pgd);
        // disjunctive.rs: 20% for precheck
        assert!((cfg.disjunctive_precheck_fraction - 0.20).abs() < 1e-6);
        // mod.rs: timeout/4 minimum for MIP
        assert!((cfg.mip_min_fraction - 0.25).abs() < 1e-6);
        // mod.rs: 5..30s MIP clamp
        assert_eq!(cfg.mip_min_secs, 5);
        assert_eq!(cfg.mip_max_secs, 30);
        // mip_highs: 5%, 0.25..2.0s
        assert!((cfg.mip_crown_ibp_fraction - 0.05).abs() < 1e-9);
        assert!((cfg.mip_crown_ibp_min_secs - 0.25).abs() < 1e-9);
        assert!((cfg.mip_crown_ibp_max_secs - 2.0).abs() < 1e-9);
        // Initial bounds: 0.20 (warmup capped to 20% of BaB budget)
        assert!((cfg.initial_bounds_fraction - 0.20).abs() < 1e-6);
        // Post-BaB PGD: 0.10 (reserve 10% for PGD fallback)
        assert!((cfg.post_bab_pgd_fraction - 0.10).abs() < 1e-6);
        // Attack extension: 0.15 of remaining (matches the historical
        // DEFAULT_EXTENSION_FRACTION in attack_extension.rs)
        assert!((cfg.attack_extension_fraction - 0.15).abs() < 1e-6);
        // Absolute phase ceilings are opt-in: unset means "pure fraction", so
        // every benchmark that does not name the knob keeps its exact behavior.
        assert!(cfg.disjunctive_pgd_max_secs.is_none());
        assert!(cfg.disjunctive_precheck_max_secs.is_none());
        // #attack-stall: the adaptive attack cutoff is opt-in too. Its A/B on
        // the population it targets (GT-unsat oval21) converted nothing, and
        // firing it where the disjunctive lane finds the sat rows costs points
        // (b61b5f10), so the sealed default must stay inert.
        assert!(cfg.disjunctive_pgd_stall_window_fraction.is_none());
        // #attack-anchor: the falsification slice stays anchored at the LEDGER
        // start by default, so every category that does not name the knob keeps
        // its exact phase arithmetic.
        assert!(!cfg.disjunctive_pgd_from_phase_start);
    }

    #[test]
    fn mip_reservation_request_requires_a_finite_nonzero_policy() {
        let mut cfg = PhaseBudgetConfig {
            mip_min_fraction: 0.0,
            mip_min_secs: 0,
            ..Default::default()
        };
        assert!(!cfg.requests_mip_reservation());

        cfg.mip_min_secs = 5;
        assert!(
            cfg.requests_mip_reservation(),
            "an explicit seconds floor independently requests a reservation"
        );
        cfg.mip_min_fraction = f32::NAN;
        assert!(
            cfg.requests_mip_reservation(),
            "a valid seconds floor remains usable when the fractional arm is non-finite"
        );

        cfg.mip_min_secs = 0;
        cfg.mip_min_fraction = 0.25;
        assert!(cfg.requests_mip_reservation());

        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.25] {
            cfg.mip_min_fraction = invalid;
            assert!(
                !cfg.requests_mip_reservation(),
                "invalid/non-positive fraction {invalid:?} must not arm a reservation"
            );
        }

        cfg.mip_min_fraction = 0.25;
        cfg.mip_max_secs = 0;
        assert!(
            !cfg.requests_mip_reservation(),
            "a zero ceiling makes every reservation zero"
        );

        cfg.mip_min_secs = 6;
        cfg.mip_max_secs = 5;
        assert!(
            !cfg.requests_mip_reservation(),
            "unordered clamps must decline instead of panicking in deadline arithmetic"
        );
    }

    #[test]
    fn serde_roundtrip_preserves_defaults() {
        let cfg = PhaseBudgetConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let deserialized: PhaseBudgetConfig = serde_json::from_str(&json).expect("deserialize");
        assert!((deserialized.upfront_pgd_fraction - cfg.upfront_pgd_fraction).abs() < 1e-6);
        assert!((deserialized.mip_crown_ibp_fraction - cfg.mip_crown_ibp_fraction).abs() < 1e-9);
    }

    #[test]
    fn partial_override_preserves_other_defaults() {
        let json = r#"{"upfront_pgd_fraction": 0.10}"#;
        let cfg: PhaseBudgetConfig = serde_json::from_str(json).expect("partial deserialize");
        assert!((cfg.upfront_pgd_fraction - 0.10).abs() < 1e-6);
        // Other fields should remain at default
        assert!((cfg.reduced_verification_fraction - 0.40).abs() < 1e-6);
        assert!((cfg.disjunctive_pgd_fraction - 0.50).abs() < 1e-6);
        assert!((cfg.post_bab_pgd_fraction - 0.10).abs() < 1e-6);
    }

    #[test]
    fn post_bab_pgd_reservation_override() {
        let json = r#"{"post_bab_pgd_fraction": 0.05}"#;
        let cfg: PhaseBudgetConfig = serde_json::from_str(json).expect("partial deserialize");
        assert!((cfg.post_bab_pgd_fraction - 0.05).abs() < 1e-6);
        // Other fields should remain at default
        assert!((cfg.upfront_pgd_fraction - 0.20).abs() < 1e-6);
        assert!((cfg.initial_bounds_fraction - 0.20).abs() < 1e-6);
    }

    /// Verifies the Duration arithmetic used in all 6 graph BaB entry points
    /// for phase_budget warmup cap. The pattern:
    ///
    /// ```text
    /// let pgd_frac = config.post_bab_pgd_fraction.clamp(0.0, 0.5);
    /// let bab_timeout = timeout.mul_f32(1.0 - pgd_frac);
    /// let frac = config.initial_bounds_fraction.clamp(0.0, 1.0);
    /// let initial_deadline = start + bab_timeout.mul_f32(frac);
    /// ```
    #[test]
    fn graph_bab_warmup_cap_duration_arithmetic() {
        use std::time::Duration;

        let timeout = Duration::from_secs(100);

        // Competitive settings: 15% warmup, 10% PGD reservation
        let cfg: PhaseBudgetConfig = serde_json::from_str(
            r#"{"initial_bounds_fraction": 0.15, "post_bab_pgd_fraction": 0.10}"#,
        )
        .expect("deserialize");

        let pgd_frac = cfg.post_bab_pgd_fraction.clamp(0.0, 0.5);
        let bab_timeout = timeout.mul_f32(1.0 - pgd_frac);
        let frac = cfg.initial_bounds_fraction.clamp(0.0, 1.0);
        let initial_budget = bab_timeout.mul_f32(frac);

        // BaB gets 90% of total (100s * 0.9 = 90s)
        assert!(
            (bab_timeout.as_secs_f64() - 90.0).abs() < 0.5,
            "bab_timeout should be ~90s, got {:.1}s",
            bab_timeout.as_secs_f64()
        );

        // Warmup gets 15% of BaB budget (90s * 0.15 = 13.5s)
        assert!(
            (initial_budget.as_secs_f64() - 13.5).abs() < 0.5,
            "initial_budget should be ~13.5s, got {:.1}s",
            initial_budget.as_secs_f64()
        );
    }

    /// Default config allocates competitive budget splits:
    /// BaB gets 90% (post_bab_pgd=0.10), warmup gets 20% of BaB budget.
    /// On a 100s timeout: warmup=18s, BaB=72s, post-BaB PGD=10s.
    #[test]
    fn graph_bab_default_config_competitive_budgets() {
        use std::time::Duration;

        let timeout = Duration::from_secs(100);
        let cfg = PhaseBudgetConfig::default();

        let pgd_frac = cfg.post_bab_pgd_fraction.clamp(0.0, 0.5);
        let bab_timeout = timeout.mul_f32(1.0 - pgd_frac);
        let frac = cfg.initial_bounds_fraction.clamp(0.0, 1.0);
        let initial_budget = bab_timeout.mul_f32(frac);

        // BaB gets 90% of total (100s * 0.9 = 90s)
        assert!(
            (bab_timeout.as_secs_f64() - 90.0).abs() < 0.5,
            "bab_timeout should be ~90s, got {:.1}s",
            bab_timeout.as_secs_f64()
        );

        // Warmup gets 20% of BaB budget (90s * 0.2 = 18s)
        assert!(
            (initial_budget.as_secs_f64() - 18.0).abs() < 0.5,
            "initial_budget should be ~18s, got {:.1}s",
            initial_budget.as_secs_f64()
        );
    }

    /// #seq-single-reserve — the measured tll two-point validation, pinned.
    ///
    /// The ledger hands the engine 483.0 s of a 600 s row. Historical
    /// (double-reserved) warmup cap: 483.0 * 0.90 * 0.85 = 369.495 s — the
    /// arithmetic validated to 0.1 s against two measured arms (372.38 s and
    /// 372.36 s walls at ~2.8 s setup; 437.44/437.46 at fraction 1.00).
    /// Corrected cap: 483.0 * 0.85 = 410.55 s. `bab_timeout` — the engine
    /// PGD fallback's window start — is 434.7 s in BOTH dispositions.
    #[test]
    fn seq_single_reserve_tll_two_point_validation() {
        use std::time::Duration;

        let mut cfg = PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.10,
            initial_bounds_fraction: 0.85,
            ..PhaseBudgetConfig::default()
        };
        let ledger_boundary = Duration::from_secs_f64(483.0);

        // Fix ON (shipped default), caller deadline present.
        assert!(
            cfg.sequential_warmup_single_reserve,
            "shipped default is ON"
        );
        let on = cfg.sequential_initial_schedule(ledger_boundary, true);
        assert!(on.warmup_single_reserve_applied);
        assert!(
            (on.warmup_cap.as_secs_f64() - 410.55).abs() < 0.05,
            "corrected cap must be 483.0 * 0.85 = 410.55 s, got {:.3}",
            on.warmup_cap.as_secs_f64()
        );
        assert!(
            (on.bab_timeout.as_secs_f64() - 434.7).abs() < 0.05,
            "the BaB search boundary must keep the engine's single reserve \
             (483.0 * 0.90 = 434.7 s), got {:.3}",
            on.bab_timeout.as_secs_f64()
        );

        // Kill switch OFF: the historical double-reserved cap, byte-identically.
        cfg.sequential_warmup_single_reserve = false;
        let off = cfg.sequential_initial_schedule(ledger_boundary, true);
        assert!(!off.warmup_single_reserve_applied);
        let historical = ledger_boundary
            .mul_f32(1.0 - cfg.post_bab_pgd_fraction.clamp(0.0, 0.5))
            .mul_f32(cfg.initial_bounds_fraction.clamp(0.0, 1.0));
        assert_eq!(
            off.warmup_cap, historical,
            "kill switch OFF must reproduce the historical formula byte-identically"
        );
        assert!(
            (off.warmup_cap.as_secs_f64() - 369.495).abs() < 0.05,
            "historical cap is 483.0 * 0.90 * 0.85 = 369.495 s, got {:.3}",
            off.warmup_cap.as_secs_f64()
        );

        // The test must be able to fail: the two dispositions must differ.
        assert_ne!(on.warmup_cap, off.warmup_cap);
        assert_eq!(on.bab_timeout, off.bab_timeout);
    }

    /// #seq-single-reserve — no caller deadline means no second reserve to
    /// decline: the config-timeout path is byte-identical with the key ON.
    #[test]
    fn seq_single_reserve_is_inert_without_a_caller_deadline() {
        use std::time::Duration;

        let cfg = PhaseBudgetConfig::default();
        assert!(cfg.sequential_warmup_single_reserve);
        let total = Duration::from_secs(100);

        let on = cfg.sequential_initial_schedule(total, false);
        let off = {
            let mut c = cfg;
            c.sequential_warmup_single_reserve = false;
            c.sequential_initial_schedule(total, false)
        };
        assert_eq!(on, off, "without a caller deadline the key must be inert");
        assert!(!on.warmup_single_reserve_applied);
        assert_eq!(on.warmup_cap, total.mul_f32(0.9).mul_f32(0.2));
    }

    /// #seq-single-reserve containment: over a grid of budgets and fractions,
    /// the corrected warmup cap never moves EARLIER than the historical cap,
    /// never past the effective total, and the BaB search boundary is
    /// bit-identical in both dispositions.
    #[test]
    fn seq_single_reserve_containment_grid() {
        use std::time::Duration;

        for total_secs in [5_u64, 20, 100, 483, 600, 1800] {
            for pgd_frac in [0.0_f32, 0.05, 0.10, 0.25, 0.5, 0.9] {
                for init_frac in [0.0_f32, 0.15, 0.20, 0.85, 1.0] {
                    let mut cfg = PhaseBudgetConfig {
                        post_bab_pgd_fraction: pgd_frac,
                        initial_bounds_fraction: init_frac,
                        ..PhaseBudgetConfig::default()
                    };
                    let total = Duration::from_secs(total_secs);
                    let on = cfg.sequential_initial_schedule(total, true);
                    cfg.sequential_warmup_single_reserve = false;
                    let off = cfg.sequential_initial_schedule(total, true);

                    assert_eq!(
                        on.bab_timeout, off.bab_timeout,
                        "the BaB boundary must not move in either disposition \
                         (total={total_secs}s pgd={pgd_frac} init={init_frac})"
                    );
                    assert!(
                        on.warmup_cap >= off.warmup_cap,
                        "the corrected cap may only move LATER \
                         (total={total_secs}s pgd={pgd_frac} init={init_frac})"
                    );
                    assert!(
                        on.warmup_cap <= total,
                        "the corrected cap must never pass the caller deadline \
                         (total={total_secs}s pgd={pgd_frac} init={init_frac})"
                    );
                }
            }
        }
    }
}
