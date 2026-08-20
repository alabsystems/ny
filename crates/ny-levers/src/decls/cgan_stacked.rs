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
