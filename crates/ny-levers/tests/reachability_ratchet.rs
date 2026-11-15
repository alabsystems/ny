// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! REACHABILITY RATCHET: how many verdict-affecting levers cannot fire in
//! competition.
//!
//! # Why this exists
//!
//! `vnncomp_scripts/run_instance.sh` — the scored entry point — exports no
//! `NY_*` variables. Every env-only gate therefore reads "unset" on a scored
//! run and its mechanism is dead code, however well it measures by hand.
//!
//! That is not a bug in any single lever. It is a MISSING STEP IN THE WORKFLOW:
//!
//!   build behind an env gate (so it can be A/B'd safely)
//!     -> run the A/B
//!     -> bank the result
//!     -> **promote to a typed preset key**        <-- never happens
//!     -> ship
//!
//! Step 4 costs five edits — a `MarginRowPreset`-style field, a reader in
//! `commands/margin_row_bab.rs`, a bridge line in `commands/vnncomp.rs`, a
//! `LeverDecl`, and a `space.rs` axis — and it has no deadline and no forcing
//! function. So it is skipped, the mechanism stays dark forever, and everyone
//! remembers it as "shipped".
//!
//! The cost of that is not hypothetical. In one session SIX mechanisms were
//! found built, correct, measured, and structurally unable to fire:
//! `NY_EPOCH_DEPTH` (the fresh-intermediates rebuild — an undeclared raw
//! `env::var` at `margin_row/mod.rs`, default 0 — BOTH HALVES SINCE PAID:
//! `a1c80f6a3` added the typed `margin_row.epoch_depth` key and the
//! 2026-08-24 ratchet conversion declared the variable and its five siblings,
//! so it is now `Deliver::PresetKey` and counted as reachable. Kept in this
//! list because it is the canonical example the gate exists for, not because
//! it is still dark), `NY_CGAN_RECLAIM` (recovers
//! 1114 s of thrown-away budget on 4 of 7 cgan rows), `NY_MARGIN_ROW_CLIP_ROWS`
//! (without it EVERY clip reader refuses, which made a banked null VACUOUS),
//! `NY_MARGIN_ROW_DOMAIN_STACK`, `margin_row.closure_adaptive` (an entire
//! runtime tier ladder, unit-tested and measured, with no bridge line), and
//! `NY_MARGIN_ROW_BETA`.
//!
//! # What this test does
//!
//! It counts axes that are `Class::VerdictAffecting` — the registry's own
//! statement that the lever can change an answer — and NOT
//! `Deliver::PresetKey`. That number may only ever go DOWN.
//!
//! There are exactly two honest ways to lower it, and both improve the tree:
//!
//! 1. PROMOTE the lever: add the typed preset key and flip `deliver` to
//!    `Deliver::PresetKey("...")`. The mechanism becomes reachable.
//! 2. RECLASSIFY it: if it is really a diagnostic or a research probe that
//!    cannot move a published bound, it does not belong in
//!    `Class::VerdictAffecting`. Move it to the honest class.
//!
//! What you may NOT do is raise the baseline. Adding a new verdict-affecting
//! env-only lever without a preset key is the exact failure this file exists to
//! stop, and it is why the number is checked rather than merely reported.
//!
//! Deliberately a RATCHET and not a hard `== 0` assertion: 62 is the measured
//! starting debt and failing the build outright would simply get the test
//! deleted. The raw-`env::var` ratchet (`tests/ratchet.rs`) has worked for
//! months on exactly this shape, including catching the author of this file
//! adding a raw read an hour before writing it.

use ny_levers::space::{Class, Deliver};

/// The checked-in baseline. LOWER IT in the same commit that promotes or
/// reclassifies a lever. It may never go up.
///
/// 2026-08-23  62  Landing measurement. 92 axes total: 62 VerdictAffecting +
///                 EnvOnly, 13 VerdictAffecting + PresetKey, 13 SafeToSearch +
///                 EnvOnly, 4 Unsafe + EnvOnly. So 83% of everything the
///                 registry says can change a verdict could not fire in the
///                 scored configuration on the day this was written.
///
/// 2026-08-23  31  #lever-reachability batch 1. Minus 31, in three kinds:
///
///                 21 PROMOTED — each got a typed preset field, a reader, the
///                 bridge line in `commands::vnncomp::run_and_translate`
///                 (`preset_gates::apply`), a setter/reader pair over an
///                 `ny_levers::gate` cell, and `Deliver::PresetKey` on its
///                 axis. The root-phase ownership lattice (3), the
///                 comprehensive-sweep sizing knobs (3), the batched
///                 intermediate refinement trio (3), the wide multi-objective
///                 pair (2), the MIP leaf oracle, the true-gradient replay,
///                 the SafeNLP cycle, the nn4sys context-group cover, and the
///                 entire attack/falsification slice (6) — which is where ny
///                 earns `sat`, the half of the score BaB cannot reach, and
///                 which was unreachable in competition in its entirety.
///
///                 4 CORRECTED — `NY_BNN_SIGN_SPACE`, `NY_BNN_STE_PGD`,
///                 `NY_UPFRONT_ATTACK_PROBE_FIRST` and `NY_MARGIN_ROW_K_ADAPT`
///                 already had a typed key, a reader AND a bridge line. The
///                 axis was wrong about them, so the gate was counting a debt
///                 that had been paid. A ratchet that miscounts in the safe
///                 direction still teaches the wrong lesson about which
///                 mechanisms are dark.
///
///                 6+1 RECLASSIFIED into `INSTRUMENT_ONLY` — the six
///                 `NY_STAR_DARK_*` controls (`run_dark_star_probe` returns
///                 `()`; `emittable_verdict_today` maps every outcome to
///                 `"unknown"`; both authority obligations undischarged) and
///                 `NY_MARGIN_ROW_GPU_SHADOW` (device pass computed, compared,
///                 DISCARDED by the M2 parity protocol — and its own
///                 configurator `NY_MARGIN_ROW_GPU_SHADOW_SITES` was already on
///                 that list, describing the shadow as "already excluded",
///                 which it was not). The rule applied, and it is the rule that
///                 keeps this gate worth having: if a lever can move a
///                 PUBLISHED BOUND it is a promotion, full stop. Stealing wall
///                 clock is not moving a bound — and `INSTRUMENT_ONLY`'s own
///                 header already says instruments are not free.
///
///                 PENDING, and this is the one coordination note: a sibling
///                 commit bridges `NY_CGAN_RECLAIM` to `bab.cgan_reclaim`.
///                 Whichever lands second sees 30 here and must lower this
///                 constant to 30 — which is exactly what
///                 `the_reachability_baseline_is_not_stale` exists to force.
///
/// 2026-08-23  30  That sibling landed second. `NY_CGAN_RECLAIM` is now
///                 `Deliver::PresetKey("bab.cgan_reclaim")`, bridged in
///                 `commands/vnncomp.rs` from `preset::cgan_reclaim_from_preset`
///                 and PROVEN reachable with a preset that sets the key and no
///                 env var (`[cgan-reclaim] ARMED (source: preset
///                 bab.cgan_reclaim)`). The set it arms now has five retention
///                 sites and recovers the whole thrown-away tail on the three
///                 cgan rows this front owns: 74/75/192 s unused -> 4 s, and
///                 `unknown` -> `timeout`, at the official 900 s budget.
///                 Promotion is still not arming: no shipped yaml sets the key.
///
///                 NOT DONE, so the next reader does not rediscover it: the
///                 18 `margin_row` axes (clip, alpha-opt, the closure ladder,
///                 beta, rebake, the GPU seam) need edits inside
///                 `crates/ny-propagate/src/margin_row/**`, which two other
///                 worktrees owned during this batch; and the 6 `ny-gpu` axes
///                 (`NY_BAB_RESNET_WIDE{,_SUBGROUP}`, `NY_SWEEP_CLASS_*`,
///                 `NY_RESNET_ERR_CONCRETIZE{,_FINE}`) need the same treatment
///                 one crate further out. Both sets are PROMOTIONS. None of the
///                 remaining axes is a reclassification candidate on the
///                 evidence available today.
///
/// 2026-08-23  20  #lever-reachability batch 2. Ten more PROMOTED, and they are
///                 the whole of the "one crate further out" list above plus the
///                 last non-`margin_row` straggler:
///
///                 the six `ny-gpu` CROWN-lane axes — `NY_BAB_RESNET_WIDE`
///                 (ships ARMED, so the key is the typed kill switch),
///                 `NY_BAB_RESNET_WIDE_SUBGROUP`, `NY_SWEEP_CLASS_ROWS`,
///                 `NY_SWEEP_CLASS_MIB`, `NY_RESNET_ERR_CONCRETIZE` and its
///                 `_FINE` refinement. Their cells live in
///                 `ny_propagate::preset_gates` because `ny-gpu` depends on
///                 `ny-propagate`: one place the bridge writes, one place a
///                 reviewer looks;
///
///                 the three `read_raw` PREDICATE gates —
///                 `NY_SEQ_WEIGHTED_BUDGET`, `NY_CGAN_STACKED_BACKWARD`,
///                 `NY_CGAN_POLISH`. These keep their own unit-tested pure
///                 predicate AT THE READ SITE and take only a cell underneath
///                 it (`ny_levers::gate::resolve_raw_predicate`), because the
///                 registry decls say in as many words that the predicate IS
///                 the spec of the arming rule, and because for
///                 `NY_SEQ_WEIGHTED_BUDGET` the declaration's parser is a
///                 DIFFERENT function: it is an opt-out on `!= "0"` declared
///                 `Text`/`Unset`, so routing it through `resolve_bool` would
///                 have answered "off" where the shipped behaviour is ON and
///                 silently disarmed the lane on every run;
///
///                 and `NY_CROWN_PARTIAL_FINITE_EXPIRY`, worth delivering
///                 precisely because the standing audit predicts a NULL unless
///                 the whole root-cause-D set switches together — a zero
///                 measured through the key is evidence about the split, and
///                 there was previously no way to measure it on a scored run.
///
///                 WHAT IS LEFT IS EXACTLY ONE THING: the 20 `margin_row` axes.
///                 Every one needs an edit inside
///                 `crates/ny-propagate/src/margin_row/**`, which two other
///                 worktrees owned throughout this session. All 20 are
///                 PROMOTIONS on the evidence available: none is a diagnostic,
///                 and none is unsafe to ship behind a default-off key.
///
/// 2026-08-24  0   #parameter-governance margin-row completion. All remaining
///                 verdict-affecting axes gained typed `margin_row.*` keys,
///                 env-over-preset resolvers, early receipt projection, and
///                 runtime delivery. The former ceiling is now a permanent
///                 zero invariant: a new verdict-affecting env-only axis is a
///                 compile-test failure, not debt the baseline may absorb.
const BASELINE_UNREACHABLE: usize = 0;

fn unreachable_verdict_affecting() -> Vec<&'static str> {
    ny_levers::space::axes()
        .iter()
        .filter(|a| a.class == Class::VerdictAffecting)
        .filter(|a| !matches!(a.deliver, Deliver::PresetKey(_)))
        .map(|a| a.name)
        .collect()
}

fn unreachable_searchable() -> Vec<&'static str> {
    ny_levers::space::axes()
        .iter()
        .filter(|axis| matches!(axis.class, Class::VerdictAffecting | Class::SafeToSearch))
        .filter(|axis| !matches!(axis.deliver, Deliver::PresetKey(_)))
        .map(|axis| axis.name)
        .collect()
}

#[test]
// `n <= BASELINE_UNREACHABLE` reduces to `n == 0` while the baseline is 0, which
// is what clippy::absurd_extreme_comparisons reports. The `<=` form is kept
// deliberately — it is the RATCHET relation and stays correct if the baseline is
// ever raised — and unlike its `>=` twin below, this direction can still FAIL for
// any n > 0, so it is awkward rather than vacuous. The allow sits on the function
// because an attribute on a macro invocation in statement position does not reach
// into the expansion (it reports `unused attribute` instead).
#[allow(clippy::absurd_extreme_comparisons)]
fn verdict_affecting_levers_reachable_in_competition_only_ever_grows() {
    let unreachable = unreachable_verdict_affecting();
    let n = unreachable.len();
    eprintln!(
        "reachability: {n} verdict-affecting levers are NOT deliverable by preset key \
         (baseline {BASELINE_UNREACHABLE}); run_instance.sh exports no NY_* vars, \
         so each of these is dead code on a scored run"
    );
    assert!(
        n <= BASELINE_UNREACHABLE,
        "REACHABILITY RATCHET TRIPPED: {n} verdict-affecting levers cannot fire in \
         competition, baseline is {BASELINE_UNREACHABLE} (+{}).\n\nA lever that \
         `Class::VerdictAffecting` says can change an answer, but that only a raw `NY_*` \
         env var can arm, is DEAD CODE on every scored run — `run_instance.sh` exports \
         no such variables. Either give it a typed preset key and set \
         `Deliver::PresetKey(..)`, or, if it genuinely cannot move a published bound, \
         reclassify it out of VerdictAffecting. Do not raise the baseline.\n\nCurrently \
         unreachable:\n{}",
        n - BASELINE_UNREACHABLE,
        unreachable
            .iter()
            .map(|s| format!("  {s}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}

/// The ratchet above is a ceiling; this is the floor that makes progress
/// visible. If the count drops, the baseline must drop with it in the same
/// commit — otherwise the gate silently stops protecting the ground that was
/// just won, which is exactly how the raw-read baseline drifted twice.
#[test]
fn the_reachability_baseline_is_not_stale() {
    let n = unreachable_verdict_affecting().len();
    // Not `n >= BASELINE_UNREACHABLE`: on a `usize` with the baseline at 0 that
    // is ALWAYS TRUE, so the floor test could not fail once the debt reached
    // zero on 2026-08-24.
    //
    // Equality carries the intent in both directions and cannot go vacuous: it
    // fails if the debt FELL below the recorded baseline (lower the constant),
    // and at today's zero baseline it also fails if the debt GREW, pinning the
    // permanent-zero invariant rather than restating an identity.
    assert_eq!(
        n, BASELINE_UNREACHABLE,
        "The reachability debt is {n} but BASELINE_UNREACHABLE is \
         {BASELINE_UNREACHABLE}. If the debt FELL, lower the constant to {n} in this \
         commit and record what was promoted or reclassified, so the gate protects the \
         new ground. If it GREW, see the ratchet message above — do not raise the \
         constant."
    );
}

/// Parameter governance is broader than verdict classification: a
/// value-neutral kernel/budget arm still cannot be optimized for competition
/// if the scored entry point has no way to deliver it. Unsafe, test-only and
/// instrumentation-only controls intentionally remain outside this set.
#[test]
fn every_competition_searchable_axis_has_typed_delivery() {
    let unreachable = unreachable_searchable();
    assert!(
        unreachable.is_empty(),
        "SEARCH DELIVERY INCOMPLETE: every VerdictAffecting and SafeToSearch axis \
         must have a typed preset route; env-only research is reserved for Unsafe, \
         test-only and instrument-only controls. Missing:\n{}",
        unreachable
            .iter()
            .map(|name| format!("  {name}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
