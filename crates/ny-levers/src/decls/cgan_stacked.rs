// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#cgan-stacked-backward`: the arming gate and the peak-bytes budget for the
//! cgan shared-prefix STACKED backward planner.
//!
//! The mechanism these two govern (docs/CGAN_STACKED_BACKWARD_2026-08-19.md,
//! resting on the measurements in docs/CGAN_COLLECTION_CACHE_DEFECTS_2026-08-03.md
//! and docs/CGAN_BOUND_QUALITY_FIX_2026-08-18.md): each demanded cgan target's
//! CROWN backward costs 95-125 s, of which 98.4-99.7 % is the SHARED upstream
//! ConvTranspose generator prefix, and 7 targets are demanded per collection —
//! about 700 s of near-duplicate walk inside a 900 s budget. Both scored graphs
//! (`cGAN_imgSz32_nCh_{1,3}.onnx`) are pure 28-node CHAINS with zero fan-out
//! tensors, so every demanded target sits on one trunk and a walk from the
//! deepest target passes through every shallower target's node. The lane plans
//! ONE dense backward walk whose seed row-concatenates every stacked member's
//! identity block, walking the shared prefix once.
//!
//! [`CGAN_STACKED_BACKWARD`] arms that lane; [`CGAN_STACKED_BUDGET_MB`] decides
//! how many members fit ONE walk, which is the only dial the design left —
//! objective chunking was investigated and deliberately not built, because
//! splitting a stacked pass into mixed-target chunks conserves the chunk count
//! and therefore the prefix re-walk count (parity, not a win). The winning form
//! is maximal rows per single walk, and that is a pure memory question.
//!
//! Why BOTH are `MoatRisk::High` even though stacking is argued to be pure
//! batching: the bit-identity claim is about the ARITHMETIC of an admitted row,
//! and every refusal falls back to the historical per-target walk, so neither
//! lever can make a bound unsound. What they DO change is which intermediate
//! bounds a scored cgan collection publishes — the stacked pass reads the
//! pre-loop bound map instead of seeing earlier targets' tightened boxes, and
//! the members it admits are the ones whose solo walks no longer run — and how
//! the collection's wall clock is spent. Converting a cgan timeout is the
//! lane's stated purpose; a lever that can do that is High by definition.
//!
//! Both are also `Bucket::Debug` with `Provenance::Unmeasured` for one blunt
//! reason: the implementing session never built or ran any of it (the serial
//! GPU pipeline owned the machine; only `rustfmt --check` was run). There is no
//! retained scored reading for either arm, and the runbook's Phase 0 exists
//! precisely to produce the first one.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const CGAN_STACKED: Scope = Scope {
    package: "ny-propagate",
    subsystem: "cgan-stacked-backward",
};

const GATE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: CGAN_STACKED,
    role: "arm the shared-prefix stacked backward lane, and with it the armed-only `[NY_CGAN_STACKED]` telemetry",
    site: "crates/ny-propagate/src/network/graph_alpha/bounds/cgan_stacked.rs:stacked_backward_enabled",
}];

const BUDGET_READERS: &[ReaderSite] = &[ReaderSite {
    scope: CGAN_STACKED,
    role: "price the single stacked walk's peak dense transients, deciding how many members are admitted",
    site: "crates/ny-propagate/src/network/graph_alpha/bounds/cgan_stacked.rs:stacked_budget_bytes",
}];

const CGAN_RECLAIM_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "cgan-budget-reclaim",
};

const CGAN_POLISH_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "cgan-survivor-polish",
};

const CGAN_POLISH_READERS: &[ReaderSite] = &[ReaderSite {
    scope: CGAN_POLISH_SCOPE,
    role: "arm the grouped-disjunctive near-miss survivor alpha polish \
           (real SPSA steps on knife-edge children; the eager pass at \
           iterations=1 performs zero optimizer steps)",
    site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/disjunctive_multi_clause.rs:cgan_polish_requested",
}];

/// One env acquisition point, four decision sites. The gate function is the
/// only reader of the name; the cross-mode collection-cache serve
/// (`graph_alpha/bounds/crown.rs`), the crown-partial expiry pair
/// (`network/ibp/crown_partial.rs:crown_partial_expiry_armed`), the
/// sequential CROWN-IBP commit-what-you-have salvage
/// (`network/ibp/crown_ibp.rs:collect_core`) and the ConvTranspose2d
/// mid-step expiry salvage
/// (`network/core/sequential/crown/patches_step.rs`) all call it.
const CGAN_RECLAIM_READERS: &[ReaderSite] = &[ReaderSite {
    scope: CGAN_RECLAIM_SCOPE,
    role: "arm the cgan budget-reclaim set: cross-mode CROWN-IBP collection-cache serve, \
           crown-partial expiry retention, sequential CROWN-IBP expiry salvage, and \
           ConvTranspose2d mid-step expiry salvage",
    site: "crates/ny-propagate/src/cgan_reclaim.rs:cgan_reclaim_armed",
}];

declare_levers! {
    registry CGAN_STACKED_LEVERS;

    /// `NY_CGAN_STACKED_BACKWARD` — exact-`"1"` master arm for the lane.
    pub CGAN_STACKED_BACKWARD = LeverDecl {
        name: "NY_CGAN_STACKED_BACKWARD",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the cgan shared-prefix STACKED backward lane in the graph-alpha CROWN \
collection loop: one dense backward walk rooted at the deepest admissible \
demanded target, whose seed carries every stacked member's identity block, so \
the shared ConvTranspose generator prefix is walked ONCE instead of once per \
demanded target. Exact `1` arms it. Absence, `0`, `true`, `01`, ` 1` and every \
other token leave it dark and the collector byte-identical to the historical \
per-target path.

THE PARSER LIVES AT THE READER AND STAYS THERE. \
`cgan_stacked::stacked_backward_enabled_from_raw` is a pure predicate pinned by \
`cgan_stacked::tests::gate_parser_is_exact_and_default_dark`; that unit test IS \
the spec of the arming rule, so only the env ACQUISITION is routed through \
`ny_levers::read_raw`, which is the same `env::var(..).ok()` lookup the reader \
used (a non-UTF-8 value reads as absent, i.e. dark, exactly as before). This \
declaration's `Bool` parser agrees with that predicate on every token: `1` arms, \
`0` is an admissible disarm, everything else is a recorded rejection resolving \
to the dark default — so the flight receipt cannot disagree with the decision \
the reader made.

WHY HIGH RISK. Stacking is pure batching and cannot make a bound unsound: an \
admitted row's arithmetic is bit-identical to its solo pass (the injection \
verifies exact zeros before writing an identity block and refuses otherwise, \
with an `injected` tripwire for a member the walk never reaches), served bounds \
flow through the same shrink-only IBP intersection, and EVERY refusal — \
planner, executor, budget, audit, fan-out — degrades to the existing \
per-target walk. What it does change is WHICH intermediate bounds a scored cgan \
collection publishes: the stacked pass sees only the pre-loop bound map rather \
than earlier targets' tightened boxes (a documented quality tradeoff, worth \
about zero on cgan where per-walk cost is flat to +/-6 % across repeats), the \
members it admits no longer run their solo walks, and the reclaimed wall clock \
is spent elsewhere. Converting a cgan timeout is the whole point of the lane, \
so it is a verdict-moving lever on the authoritative route.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark exact-one gate; the implementing session built and ran \
                     NOTHING (docs/CGAN_STACKED_BACKWARD_2026-08-19.md section 1 records \
                     rustfmt-only verification), so no arm has a retained scored reading",
        },
        owner: CGAN_STACKED,
        readers: GATE_READERS,
    };

    /// `NY_CGAN_RECLAIM` — exact-`"1"` master arm for the cgan budget-reclaim
    /// set (docs/CGAN_BUDGET_RECLAIM_2026-08-20.md). Ships DARK.
    pub CGAN_RECLAIM = LeverDecl {
        name: "NY_CGAN_RECLAIM",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the cgan BUDGET-RECLAIM set (docs/CGAN_BUDGET_RECLAIM_2026-08-20.md). The \
diagnosed mechanism (scratchpad cgan_live_route.md, 2026-08-19): the 900 s \
grouped-disjunctive-eager route is a CLOSING tree starved of budget — ~68 % of \
the wall is spent before the first split because (a) the CLI per-output \
precheck's ~385 s Standard-mode CROWN-IBP collection never serves the \
in-verifier CganComplete root re-collection (fingerprint differs ONLY in the \
collection-mode tag byte), and (b) the crown-partial deadline-PRESENCE pair \
discards its optimized routes on every scored run (saved=0 discarded=32).

Exact `1` arms three sites, all in ny-propagate:
1. CROSS-MODE COLLECTION-CACHE SERVE (graph_alpha/bounds/crown.rs): a typed \
cGAN consumer (CganComplete / CganSparseTargetComplete) carrying its own \
certified forward-linear baseline may be served a COMPLETE same-box \
Standard-mode entry. The serve is shrink-only intersected against that \
baseline via the bug-19 `census_commit::shrink_only_commit` machinery, so the \
typed-mode 'never wider than the baseline' publication contract survives \
(disjoint f32 corners fail closed to the served certified value and are \
counted). Truncated entries never serve across modes.
2. CROWN-PARTIAL EXPIRY RETENTION (network/ibp/crown_partial.rs): arms the same \
root-cause-D expiry semantics as NY_CROWN_PARTIAL_FINITE_EXPIRY for the GPU \
partial route + sparse seed pair. Route selection only; the sound-GPU gate \
(default ON) still forbids any unsound GPU backward from deciding a bound.
3. SEQUENTIAL CROWN-IBP EXPIRY SALVAGE (network/ibp/crown_ibp.rs): when the \
collection's global deadline expires mid-walk, commit-what-you-have — the \
already-completed per-node tightenings (each already shrink-only intersected \
against its incumbent IBP bound) are returned with the remaining layers filled \
from the sound IBP map and marked DeadlineExceeded, instead of discarding the \
whole map with an Err.
4. CONVTRANSPOSE2D MID-STEP EXPIRY SALVAGE \
(network/core/sequential/crown/patches_step.rs, the ConvTranspose2d Patches \
arm): the Anchored planners poll one node deadline from admission through \
publication, so a deadline crossing DURING the walk returns Err(DeadlineExceeded) \
from the leaf. The sequential walk takes that with `?` and it escapes to the CLI \
as 'Verification produced no verdict (sound unknown)' with up to a third of the \
instance budget unspent (measured: cGAN_imgSz64_nCh_3 prop_1/prop_3 stop at \
826/825 s of 900 s). Armed, the arm returns CrownStepResult::IbpFallback \
instead, i.e. the walk discards its partial CROWN relation and keeps the \
enclosure it already holds. It manufactures no value: the same walk already \
returns that enclosure when it finds the deadline expired at the step's ENTRY \
check one branch earlier. Its R9 observable, the [convtranspose-expiry] line, \
is UNCONDITIONAL and names the lane in both arms.
5. FORWARD-FALLBACK PUBLICATION PAST EXPIRY (graph_crown/propagation.rs \
plain_forward_fallback, core/sequential/crown.rs clone_crown_forward_fallback). \
Item 4's partner, and it is REQUIRED for item 4 to be worth anything: with item \
4 alone the absorbed walk asks for its sound forward enclosure and the CLONE of \
that enclosure is itself deadline-refused, so the row dies one link later \
(MEASURED, cGAN_imgSz64_nCh_3 prop_1 at 900 s: stopper moves from \
'ConvTranspose2d Patches backward: deadline exceeded' at 826 s to 'GraphNetwork \
DAG-CROWN: deadline exceeded before forward fallback clone' at 825 s). By the \
time these clones run the enclosure is already computed and already resident \
and the only remaining work is copying 2*len f32 endpoints (230 KB on cgan's \
widest node), so the refusal buys no wall clock and costs the whole instance. \
Armed, the wall-clock refusal is lifted for that copy alone: every memory \
receipt inside the clone is untouched, the copy is bit-exact, and no search \
gets extra time.

Every arm is a monotone intersect of independently certified enclosures, a \
route selection between two sound arms, or a retention of already-certified \
bounds; no arm manufactures a bound value. High risk because it changes which \
intermediate bounds a scored cgan run publishes and how its wall clock is \
spent — converting a cgan timeout is its stated purpose.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark exact-one gate; the pre-registered A/B \
                     (docs/CGAN_BUDGET_RECLAIM_2026-08-20.md: official row 2, 3 runs/arm \
                     at the 900 s budget) has not run yet, so no arm has a retained \
                     scored reading. Engagement observables are pre-declared: \
                     [cgan-reclaim] serve/salvage lines, best_gap trajectory, and \
                     crown-partial-salvage saved>0",
        },
        owner: CGAN_RECLAIM_SCOPE,
        readers: CGAN_RECLAIM_READERS,
    };

    /// `NY_CGAN_POLISH` — exact-`"1"` arm for the grouped-disjunctive
    /// near-miss survivor alpha polish. Ships DARK.
    pub CGAN_POLISH = LeverDecl {
        name: "NY_CGAN_POLISH",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the NEAR-MISS SURVIVOR ALPHA POLISH on the grouped-disjunctive EAGER \
input-split lane (cgan's 900 s route, `route=grouped-disjunctive-eager`). \
Exact `1` arms it; absence, `0`, `true`, `01`, ` 1` and every other token \
leave it dark and the lane byte-identical.

THE DIAGNOSED MECHANISM (docs/BIG3_CORE_CAMPAIGN_2026-08-19.md, cgan \
asymptote section): cgan row 2 converges to best_gap -2e-4..-3e-4 \
reproducibly while expansions are NOT binding (fraction probe: +350 \
expansions moved the bound 1e-4). The residual is bound quality, and the \
warm-alpha loop's per-child refinement runs `input_split_alpha_iteration: 1` \
— which performs ZERO SPSA optimizer steps (the optimization loop breaks at \
`iter == iterations-1` BEFORE the first gradient), so the entire BaB tree is \
bounded with the root's frozen alphas re-masked per child.

Armed, a child that survives its eager bound INSIDE the knife-edge band \
(worst-clause gap > -5e-3) is re-bounded ONCE through the SAME certified \
warm-alpha machinery (`compute_warm_start_crown_bounds_with_refined_alpha`) \
with 3 iterations = 2 real SPSA/Adam steps, floor-merged into the running \
component-wise lower-bound maximum, and its refined alphas seed its own \
children. Children outside the band keep the full-speed 1-iteration pass.

SOUND: every polish iterate keeps alpha in [0,1], each bound is a complete \
CROWN-with-IBP-fallback pass over the child's exact sub-box, and the \
running floor is a component-wise max of sound lower bounds — the polish \
can only tighten a child, never over-claim; any polish error keeps the \
already-computed sound pass-0 bounds. High risk because converting a cgan \
timeout on the scored route is its stated purpose, and in-band children \
cost ~3.5x their pass-0 bound time (which is the intended trade: expansions \
are measured non-binding on the target row).",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark exact-one gate; the pre-registered armed measurement \
                     (official cgan row 2, 900 s, vs the banked NY_CGAN_RECLAIM-only \
                     3/3 best_gap=-3e-4 baseline) has not run yet. Engagement \
                     observables are pre-declared: the [cgan-polish] ARMED line, \
                     status=progress/final attempt counters with gap_moved_sum, and \
                     the [disjunctive-trajectory] best_gap trajectory",
        },
        owner: CGAN_POLISH_SCOPE,
        readers: CGAN_POLISH_READERS,
    };

    /// `NY_CGAN_STACKED_BUDGET_MB` — peak-bytes budget for the single walk, in
    /// whole MiB.
    pub CGAN_STACKED_BUDGET_MB = LeverDecl {
        name: "NY_CGAN_STACKED_BUDGET_MB",
        kind: LeverKind::U64,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Overrides the peak-bytes budget the stacked planner prices its single walk \
against, in whole MiB. The planner admits members greedily deepest-first while \
the projected peak (`dense_pair_bytes(total_rows, max_walk_width) * 3`, for the \
A-pair, the certified-error pair and one step transient) stays within this \
budget; members that do not fit stay on the existing per-target path.

PARSER, PRESERVED VERBATIM: `parse::<usize>()` with NO trim, so a padded \
` 24000` is REJECTED and leaves the shipped budget — the same non-trimming \
contract as `NY_MARGIN_ROW_CLIP_TOPK`. Declared `U64` rather than `UsizeTrimmed` \
for exactly that reason; the reader keeps a `usize::try_from`, so on a 64-bit \
target the behaviour is identical and on a 32-bit one an above-`usize` value is \
refused by the reader and resolves to the same shipped budget (only the receipt \
records it as accepted rather than rejected).

`Unset`, not a number, because ABSENCE means the host-adaptive CPU dense budget \
applies (`cpu_crown_dense_budget_bytes`, 2 GiB floor) — not a budget of zero. \
Explicit `0` is admissible and is a real, different setting: the chokepoint \
hands the zero back, the MiB->bytes multiplication happens AT THE READER, and a \
zero-byte budget puts every member over budget so the lane declines. Overflow of \
that multiplication also falls back to the host budget, and that fallback stays \
at the reader too.

WHY HIGH RISK. It cannot arm the lane — with `NY_CGAN_STACKED_BACKWARD` dark it \
is read only by a planner that never runs — and it cannot make a bound unsound, \
since an over-large stack is refused by the executor's existing mid-walk densify \
guards and every refusal falls back to the per-target walk. But on an armed run \
it decides HOW MANY members share one walk, therefore which targets get their \
intermediate bounds from the stacked pass instead of a solo (possibly chunked) \
walk and how much collection wall is reclaimed: at the default budget the plan \
stacks 2-4 discriminator targets, at 24 GB it stacks 4-5, and the arithmetic \
says all 7 would need ~37 GB and is never admissible. The runbook additionally \
flags the HOST hazard that made this an override rather than a raised default: \
on the GB10's unified memory a 24 GB stacked walk shares memory with the host \
and must not push the process into the OOM killer.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark research override, inert while its master gate is dark, which is \
                     the shipped state; the peak-bytes model behind it is an ARITHMETIC \
                     projection with no measured RSS comparison yet — Phase 0 of the \
                     runbook exists to take that first reading",
        },
        owner: CGAN_STACKED,
        readers: BUDGET_READERS,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_with, LeverValue, Source};

    fn resolve(decl: &'static LeverDecl, raw: Option<&str>) -> (LeverValue, Source) {
        let resolved = read_with(decl, |_| raw.map(str::to_owned));
        (resolved.value, resolved.source)
    }

    /// The declaration must agree with the reader's pure predicate
    /// (`stacked_backward_enabled_from_raw`, whose own test uses exactly these
    /// tokens) on EVERY token — otherwise the receipt would record an arming
    /// decision the run did not make.
    #[test]
    fn gate_agrees_with_the_readers_exact_one_predicate() {
        assert_eq!(
            resolve(&CGAN_STACKED_BACKWARD, None),
            (LeverValue::Bool(false), Source::Default)
        );
        assert_eq!(
            resolve(&CGAN_STACKED_BACKWARD, Some("1")),
            (LeverValue::Bool(true), Source::LegacyEnv)
        );
        assert_eq!(
            resolve(&CGAN_STACKED_BACKWARD, Some("0")),
            (LeverValue::Bool(false), Source::LegacyEnv)
        );
        for raw in ["", "true", "01", " 1", "2"] {
            let (value, source) = resolve(&CGAN_STACKED_BACKWARD, Some(raw));
            assert_eq!(value, LeverValue::Bool(false), "raw={raw:?}");
            assert_eq!(source, Source::LegacyEnvRejected, "raw={raw:?}");
        }
    }

    /// The budget parser does NOT trim; that is the reader's contract verbatim.
    #[test]
    fn budget_rejects_padding_and_leaves_the_host_budget() {
        assert_eq!(
            resolve(&CGAN_STACKED_BUDGET_MB, Some("24000")),
            (LeverValue::U64(24_000), Source::LegacyEnv)
        );
        for raw in [" 24000", "24000 ", "24_000", "lots", "-1"] {
            let (value, source) = resolve(&CGAN_STACKED_BUDGET_MB, Some(raw));
            assert_eq!(value, LeverValue::Unset, "raw={raw:?}");
            assert_eq!(source, Source::LegacyEnvRejected, "raw={raw:?}");
        }
        assert_eq!(
            resolve(&CGAN_STACKED_BUDGET_MB, None),
            (LeverValue::Unset, Source::Default)
        );
    }

    /// The polish declaration must agree with the reader's exact-one predicate
    /// (`cgan_polish_token_arms` in `disjunctive_multi_clause.rs`) on every
    /// token, mirroring the CGAN_RECLAIM contract.
    #[test]
    fn polish_gate_agrees_with_the_readers_exact_one_predicate() {
        assert_eq!(
            resolve(&CGAN_POLISH, None),
            (LeverValue::Bool(false), Source::Default)
        );
        assert_eq!(
            resolve(&CGAN_POLISH, Some("1")),
            (LeverValue::Bool(true), Source::LegacyEnv)
        );
        assert_eq!(
            resolve(&CGAN_POLISH, Some("0")),
            (LeverValue::Bool(false), Source::LegacyEnv)
        );
        for raw in ["", "true", "01", " 1", "2"] {
            let (value, source) = resolve(&CGAN_POLISH, Some(raw));
            assert_eq!(value, LeverValue::Bool(false), "raw={raw:?}");
            assert_eq!(source, Source::LegacyEnvRejected, "raw={raw:?}");
        }
    }

    #[test]
    fn polish_is_dark_debug_and_high_risk() {
        assert_eq!(CGAN_POLISH.bucket, Bucket::Debug);
        assert_eq!(CGAN_POLISH.moat, MoatRisk::High);
        assert!(matches!(
            CGAN_POLISH.provenance,
            Provenance::Unmeasured { .. }
        ));
        assert!(matches!(CGAN_POLISH.default, DefaultSpec::Bool(false)));
    }

    /// Explicit zero is a real setting (a zero-byte budget declines the lane)
    /// and must reach the reader as `0`, not collapse into absence.
    #[test]
    fn explicit_zero_budget_is_handed_back_to_the_reader() {
        assert_eq!(
            resolve(&CGAN_STACKED_BUDGET_MB, Some("0")),
            (LeverValue::U64(0), Source::LegacyEnv)
        );
    }

    #[test]
    fn both_levers_are_dark_debug_and_high_risk() {
        for decl in [&CGAN_STACKED_BACKWARD, &CGAN_STACKED_BUDGET_MB] {
            assert_eq!(decl.bucket, Bucket::Debug, "{}", decl.name);
            assert_eq!(decl.moat, MoatRisk::High, "{}", decl.name);
            assert!(
                matches!(decl.provenance, Provenance::Unmeasured { .. }),
                "{}",
                decl.name
            );
        }
        assert!(matches!(
            CGAN_STACKED_BACKWARD.default,
            DefaultSpec::Bool(false)
        ));
        assert!(matches!(CGAN_STACKED_BUDGET_MB.default, DefaultSpec::Unset));
    }
}
