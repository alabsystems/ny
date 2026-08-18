// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#comprehensive-rows-probe`: the measurement overrides for the root
//! comprehensive intermediate sweep.
//!
//! These four exist to answer ONE open design question, and they are grouped
//! because none of them means anything alone. The sweep is memory-bound in ROWS
//! — 1.4 GB peak at 144 rows, which is 0.26% coverage of ~55,000 eligible
//! neurons — and partial coverage is already known not to convert (top-3 = 69%
//! of width, still `verified 0/99`). So the question is the SCALING LAW: is that
//! 1.4 GB mostly fixed overhead or marginal per-row cost? One answer implies a
//! single wide sweep, the other implies row-chunked accumulation, and a single
//! data point cannot distinguish them.
//!
//! [`SWEEP_CLASS_MIB`] and [`SWEEP_CLASS_ROWS`] widen one sweep;
//! [`INTERM_ROW_CHUNKS`] and [`ROOT_COMP_GPU_INTERM_SECS`] buy coverage with
//! repetition instead; [`ROOT_COMP_GPU_INTERM_ROWS`] raises the per-target cap
//! that binds either way.
//!
//! NONE OF THEM CAN MAKE A BOUND UNSOUND, and the reason is structural rather
//! than a promise: every chunk is atomic over its own window, every window is
//! cut from ONE frozen transcript so the windows stay disjoint as earlier chunks
//! tighten the live map, every commit is a shrink-only intersect, and the
//! backend still validates the whole typed request against `max_device_bytes`
//! and the deadline. Stopping early — deadline, exhaustion, or a mid-run decline
//! — simply leaves fewer chunks applied, and each applied chunk is
//! independently valid. An over-large request is REFUSED by the backend, which
//! is itself the signal being measured.
//!
//! They are nonetheless all `MoatRisk::High` and all dark, for a reason that has
//! nothing to do with bound soundness: on a unified-memory part the device
//! memory is shared with the host, and an earlier over-allocation here caused a
//! GLOBAL OOM. Shipping any of these on requires the device-class policy to be
//! revisited for unified memory as its own justified change.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const ROOT_SWEEP: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-comprehensive-interm-sweep",
};

const GPU_SWEEP_CARRIER: Scope = Scope {
    package: "ny-gpu",
    subsystem: "crown-backward-sweep-carrier",
};

const INTERM_ROW_CHUNKS_READERS: &[ReaderSite] = &[ReaderSite {
    scope: ROOT_SWEEP,
    role: "cap how many disjoint row windows the sweep accumulates; 1 is the shipped single-sweep behaviour, byte-identical",
    site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/intermediate_sweep.rs",
}];

const ROOT_COMP_ROWS_READERS: &[ReaderSite] = &[ReaderSite {
    scope: ROOT_SWEEP,
    role: "raise the absolute per-target row cap for measurement",
    site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/root.rs:root_comprehensive_gpu_interm_rows_override",
}];

const ROOT_COMP_SECS_READERS: &[ReaderSite] = &[ReaderSite {
    scope: ROOT_SWEEP,
    role: "raise the phase's local authority slice, trading budget from elsewhere for coverage",
    site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/root.rs:root_comprehensive_gpu_interm_secs_override",
}];

const SWEEP_CLASS_READERS: &[ReaderSite] = &[ReaderSite {
    scope: GPU_SWEEP_CARRIER,
    role: "override the device-class memory/row proxy that misjudges unified-memory parts",
    site: "crates/ny-gpu/src/wgpu_device/ops/crown_backward.rs",
}];

declare_levers! {
    registry COMPREHENSIVE_ROWS_LEVERS;

    /// `NY_INTERM_ROW_CHUNKS` — how many disjoint row windows to accumulate.
    pub INTERM_ROW_CHUNKS = LeverDecl {
        name: "NY_INTERM_ROW_CHUNKS",
        kind: LeverKind::UsizeTrimmed,
        // 1, not Unset: the reader's `unwrap_or(1)` means absent has a real
        // shipped VALUE — exactly one sweep — rather than "feature off".
        default: DefaultSpec::U64(1),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Caps the number of disjoint row windows the comprehensive intermediate sweep \
runs and accumulates. The parser is `trim().parse::<usize>()` filtered on \
`> 0`, preserved exactly, with absent or malformed leaving 1.

1 IS THE SHIPPED PATH AND IS BYTE-IDENTICAL to the pre-chunking single sweep, \
which is why the default is a value rather than Unset. Above 1 the sweep trades \
TIME for coverage at constant peak device memory: each window is atomic, all \
windows are cut from one frozen transcript so they stay disjoint even as \
earlier chunks tighten the live bounds, and each commit is a shrink-only \
intersect. Measured: four ~4.3 s chunks at the official 100 s budget give 512 \
rows/target and a root census of 82/99 on `idx_2132`, against 0/99 unchunked.",
        provenance: Provenance::Unmeasured {
            why_ok: "the shipped default reproduces the single-sweep path exactly; \
                     no armed-vs-unarmed scored-row comparison has been retained, and \
                     the 82/99 census figure is a coverage observation, not a verdict conversion",
        },
        owner: ROOT_SWEEP,
        readers: INTERM_ROW_CHUNKS_READERS,
    };

    /// `NY_ROOT_COMP_GPU_INTERM_ROWS` — raise the per-target row cap.
    pub ROOT_COMP_GPU_INTERM_ROWS = LeverDecl {
        name: "NY_ROOT_COMP_GPU_INTERM_ROWS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Raises the absolute per-target row cap for the comprehensive root sweep. \
`trim().parse::<usize>()` filtered on `> 0`; absent or malformed leaves the \
shipped cap (32, retried down to 16 on this device profile — 144 rows against \
~55,000 eligible neurons).

Unset rather than a number because absence means the SHIPPED cap applies, not \
that a cap of zero does. Raising it cannot make a bound unsound: the backend \
still validates the whole typed request, still honours `max_device_bytes` and \
the deadline, and the host still commits shrink-only. An over-large request is \
refused by the backend — and that refusal is precisely the scaling-law signal \
this lever exists to collect.",
        provenance: Provenance::Unmeasured {
            why_ok: "measurement-only and dark; its purpose is to produce the second \
                     data point that would let the memory-vs-rows scaling law be stated \
                     at all, so by construction there is nothing measured to cite yet",
        },
        owner: ROOT_SWEEP,
        readers: ROOT_COMP_ROWS_READERS,
    };

    /// `NY_ROOT_COMP_GPU_INTERM_SECS` — raise the phase's authority slice.
    pub ROOT_COMP_GPU_INTERM_SECS = LeverDecl {
        name: "NY_ROOT_COMP_GPU_INTERM_SECS",
        kind: LeverKind::U64Trimmed,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Raises the comprehensive sweep phase's local authority slice, in whole seconds. \
`trim().parse::<u64>()` filtered on `> 0`; absent or malformed leaves the \
shipped 20 s slice.

With row-chunking the sweep trades time for coverage, so at the official 100 s \
budget this slice is what BINDS — only four ~4.3 s chunks fit. Raising it \
cannot make a bound unsound (every chunk stays atomic, deadline-bounded and \
shrink-only); it can only spend more of the budget here and less elsewhere, \
which is exactly the trade being measured. Note the related knob that was \
REJECTED as a hack and must not return: raising the phase's SHARE of the global \
budget (`bounded_root_crown_interm_deadline`'s 0.5) was reverted in `76a86e7a`, \
because it only moves wall clock away from BaB, measures 0/10 on cifar100, and \
`add_time_bonus=False` makes reclaimed wall worth nothing.",
        provenance: Provenance::Unmeasured {
            why_ok: "measurement-only and dark; whether more slice converts an 82/99 \
                     census into a root proof is the open question, so no conversion \
                     evidence exists to cite",
        },
        owner: ROOT_SWEEP,
        readers: ROOT_COMP_SECS_READERS,
    };

    /// `NY_SWEEP_CLASS_MIB` — override the device-class memory proxy.
    pub SWEEP_CLASS_MIB = LeverDecl {
        name: "NY_SWEEP_CLASS_MIB",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Overrides the sweep carrier's device-class memory budget, in MiB. \
`trim().parse::<usize>()` filtered on `> 0`; absent or malformed leaves the \
shipped class policy. Independent of `NY_SWEEP_CLASS_ROWS` — setting either \
alone overrides only that half.

WHY IT EXISTS: the class table is a proxy for \"how much memory does this device \
class usually have\", and on a unified-memory part it is wrong in one direction \
— an `IntegratedGpu` is given 2 GiB on a board with 121 GiB shared.

WHY IT IS NOT SIMPLY RAISED: that 121 GiB is shared WITH THE HOST, and an \
earlier over-allocation here caused a global OOM — the machine, not the run. \
The backend preflight still computes exact simultaneous liveness and refuses \
what it cannot honour, so a bound stays sound either way; what is at risk is the \
host. Shipping a better policy needs the device-class table revisited for \
unified memory as its own justified change, not this override left armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "measurement scaffolding, dark by default; the shipped class policy \
                     is what every scored run uses and this has no retained A/B",
        },
        owner: GPU_SWEEP_CARRIER,
        readers: SWEEP_CLASS_READERS,
    };

    /// `NY_SWEEP_CLASS_ROWS` — override the device-class row budget.
    pub SWEEP_CLASS_ROWS = LeverDecl {
        name: "NY_SWEEP_CLASS_ROWS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Overrides the sweep carrier's device-class row budget. \
`trim().parse::<usize>()` filtered on `> 0`; absent or malformed leaves the \
shipped class policy (16 rows on this integrated profile, which is 0.26% \
coverage of the eligible neurons — the sweep completes atomically and still \
moves the root census by nothing). Independent of `NY_SWEEP_CLASS_MIB`; see \
that declaration for why raising either is a host-safety question rather than a \
soundness one.",
        provenance: Provenance::Unmeasured {
            why_ok: "measurement scaffolding, dark by default; the shipped class policy \
                     is what every scored run uses and this has no retained A/B",
        },
        owner: GPU_SWEEP_CARRIER,
        readers: SWEEP_CLASS_READERS,
    };

    /// `NY_ROOT_OBJECTIVE_DIRECTED_ROWS` — exact-zero rollback for
    /// objective-directed row ranking, which SHIPS ON.
    pub ROOT_OBJECTIVE_DIRECTED_ROWS = LeverDecl {
        name: "NY_ROOT_OBJECTIVE_DIRECTED_ROWS",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Rollback for objective-directed row ranking in the comprehensive root sweep. \
The FEATURE is default-on: the scored entry point exports exactly one `NY_*` \
variable, so an env-gated improvement cannot fire in competition however well it \
measures. This lever exists only to hold the disarmed A/B arm. Exact `0` \
disarms and restores the historical width ordering byte for byte; absent, exact \
`1`, malformed UTF-8 and non-UTF-8 values all leave it armed.

Ranking is advisory — it reorders a bounded row budget and selects nothing \
unsound — but reordering changes WHICH pre-activations get tightened and \
therefore which intermediate bounds are published, so it is High-risk on the \
authoritative route rather than None.

NOTE: this declaration TIGHTENS the parser. The raw reader it replaced disarmed \
on `\"0\"` OR `\"false\"`; `\"false\"` is now rejected and leaves the feature armed, \
matching `NY_GRAPH_MIP_LEAF_SAT` and every other exact-zero kill switch in this \
registry (the chokepoint deliberately refuses `true`/`false`/`yes` spellings).",
        provenance: Provenance::Unmeasured {
            why_ok: "the lever itself is the dark rollback arm, not the shipped \
                     behaviour; its introducing commit 78184c86e is titled \
                     UNVALIDATED and retains no scored-row A/B, so nothing is \
                     claimed for either arm here",
        },
        owner: ROOT_SWEEP,
        readers: &[ReaderSite {
            scope: ROOT_SWEEP,
            role: "hold the disarmed arm for the objective-directed row ranking A/B",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/\
                   batched/root_phases.rs:objective_directed_rows_enabled",
        }],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_with, LeverValue, Source};

    #[test]
    fn chunk_count_defaults_to_the_byte_identical_single_sweep() {
        let resolved = read_with(&INTERM_ROW_CHUNKS, |_| None);
        assert_eq!(resolved.value, LeverValue::U64(1));
        assert_eq!(resolved.source, Source::Default);
    }

    #[test]
    fn measurement_overrides_preserve_the_trimming_positive_parser() {
        for decl in [
            &INTERM_ROW_CHUNKS,
            &ROOT_COMP_GPU_INTERM_ROWS,
            &ROOT_COMP_GPU_INTERM_SECS,
            &SWEEP_CLASS_MIB,
            &SWEEP_CLASS_ROWS,
        ] {
            // Whitespace is tolerated, exactly as every one of these readers'
            // `trim().parse()` did before the migration.
            let trimmed = read_with(decl, |_| Some(" 12 ".to_owned()));
            assert_eq!(trimmed.value, LeverValue::U64(12), "{}", decl.name);
            assert_eq!(trimmed.source, Source::LegacyEnv, "{}", decl.name);

            // A malformed value is a RECORDED rejection that leaves the shipped
            // default, never a silent zero.
            let malformed = read_with(decl, |_| Some("lots".to_owned()));
            assert_eq!(malformed.source, Source::LegacyEnvRejected, "{}", decl.name);
        }
    }

    /// The `> 0` filters stay at their readers, so the chokepoint must still
    /// hand `0` back rather than swallowing it — otherwise the reader could not
    /// tell "explicitly zero" from "absent" and the two mean different things.
    #[test]
    fn zero_resolves_and_is_left_for_the_readers_to_filter() {
        let resolved = read_with(&SWEEP_CLASS_ROWS, |_| Some("0".to_owned()));
        assert_eq!(resolved.value, LeverValue::U64(0));
        assert_eq!(resolved.source, Source::LegacyEnv);
    }

    #[test]
    fn every_override_is_dark_and_high_risk() {
        for decl in [
            &INTERM_ROW_CHUNKS,
            &ROOT_COMP_GPU_INTERM_ROWS,
            &ROOT_COMP_GPU_INTERM_SECS,
            &SWEEP_CLASS_MIB,
            &SWEEP_CLASS_ROWS,
        ] {
            assert_eq!(decl.bucket, Bucket::Debug, "{}", decl.name);
            assert_eq!(decl.moat, MoatRisk::High, "{}", decl.name);
            assert!(
                matches!(decl.provenance, Provenance::Unmeasured { .. }),
                "{}",
                decl.name
            );
        }
    }
}
