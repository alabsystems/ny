// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Two BaB diagnostic gates that arrived as raw reads (#bab-diagnostics).
//!
//! Both are INSTRUMENTS, not treatments: they add stderr and clocks and change
//! no bound. They are declared because the ratchet forbids a raw
//! `env::var("NY_...")`, and because their two parsers disagree with each other
//! in a way worth recording — one is presence-armed, the other exact-`"1"`.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const BAB_DIAG: Scope = Scope {
    package: "ny-propagate",
    subsystem: "bab-diagnostics",
};

const ACASXU_PROF_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: BAB_DIAG,
        role: "arm the per-phase BaB profile print on the sequential engine core",
        site: "crates/ny-propagate/src/beta_crown/engine/core.rs (prof)",
    },
    ReaderSite {
        scope: BAB_DIAG,
        role: "arm the initial-verify phase profile print",
        site: "crates/ny-propagate/src/beta_crown/engine/verify_phases/initial.rs (bab_prof)",
    },
    ReaderSite {
        scope: BAB_DIAG,
        role: "arm the multi-objective verify profile print",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/verify.rs (prof)",
    },
];

const CHILD_TRACE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BAB_DIAG,
    role: "OnceLock-latched gate for the per-child BaB trace line",
    site: "crates/ny-propagate/src/beta_crown/engine/domain/child.rs (child trace latch)",
}];

declare_levers! {
    registry BAB_DIAGNOSTIC_LEVERS;

    /// `NY_ACASXU_PROF` — PRESENCE-armed BaB phase profiling.
    pub ACASXU_PROF = LeverDecl {
        name: "NY_ACASXU_PROF",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
PRESENCE-armed, and that is the point: all three readers test `is_ok()`, so \
`NY_ACASXU_PROF=0` and even `NY_ACASXU_PROF=` ARM it. Declared as Text with \
`DefaultSpec::Unset` rather than rounded to a Boolean precisely so the receipt \
records that spelling — a Bool declaration would imply `0` disarms, which is \
false at every one of its sites (the `NY_MIP_TRACE` precedent).

Arms the per-phase BaB profile print (root census, phase walls). It is the \
instrument the campaign uses to read the root census in a single run. \
MoatRisk::Low because it cannot move a bound — but it is NOT free: it adds \
clocks and stderr to a deadline-sensitive run, so it must never appear in a \
timing arm.",
        provenance: Provenance::Unmeasured {
            why_ok: "diagnostic instrument, unset by default, prints only; no shipped \
                     path reads it and no bound depends on it",
        },
        owner: BAB_DIAG,
        readers: ACASXU_PROF_READERS,
    };

    /// `NY_BAB_CHILD_TRACE` — exact-`"1"` per-child BaB trace.
    pub BAB_CHILD_TRACE = LeverDecl {
        name: "NY_BAB_CHILD_TRACE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Exact-`\"1\"` gate for the per-child BaB trace line, latched once in a \
`OnceLock` so it cannot be varied within a process. Note the deliberate \
contrast with `NY_ACASXU_PROF` in this same module: that one is \
presence-armed, this one is not. Two BaB diagnostics, two parsers — the \
declarations exist so nobody has to guess which is which.

MoatRisk::Low: print-only, no bound effect. Per-child printing is high-volume, \
so like every instrument here it must stay out of timing arms.",
        provenance: Provenance::Unmeasured {
            why_ok: "print-only diagnostic, dark by default; no verdict path consumes it",
        },
        owner: BAB_DIAG,
        readers: CHILD_TRACE_READERS,
    };
}
