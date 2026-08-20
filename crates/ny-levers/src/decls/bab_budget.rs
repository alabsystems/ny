// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#bab-floor`: the root-window arbitration for the multi-objective graph
//! root pipeline.
//!
//! ## What these exist to fix
//!
//! `evaluate_root_borrowed` receives one deadline — the whole BaB window — and
//! hands the SAME deadline to every root phase. Each phase then claims
//! `min(fixed_cap, k x whatever remains)` of it: the alpha bootstrap a fixed
//! 40 s, the comprehensive intermediate sweep `min(20 s, 0.5 x remaining)`, the
//! sparse sweep <= 8 s, the dense head 2 s, the root objective pass a 3 s
//! grace. Nothing subtracts a slice for branch-and-bound, so BaB is the
//! RESIDUE — and on cifar100_2024 idx_2176 at the official 100 s budget the
//! residue was zero: `effective-bab=72.892s` was printed, the ladder spent
//! ~63.5 s of it, and the BaB loop was never entered (`NY_PHASE_TELEMETRY=1`
//! shows `root-objective start` and then no `root-objective end`, no
//! `multiobj-bab-ready`, and no BaB-side marker at all).
//!
//! A per-phase fraction cannot fix this. Ten claimants each taking half of
//! what is left still converge on the whole window, because the leftover is a
//! PRODUCT of fractions rather than a reservation.
//!
//! ## What they do
//!
//! [`BAB_RESERVE_FRAC`] names BaB's share of the root window and subtracts it
//! FIRST; [`ROOT_SPEC_FRAC`] does the same for the root objective pass; and
//! [`ROOT_ALPHA_FRAC`] converts the bootstrap's fixed 40 s wall into a share of
//! the same window. Whatever is left is what the sweeps divide, using their
//! existing `k x remaining` arithmetic unchanged — they simply divide a smaller
//! remainder. The split itself is [`ny_core::phase_window::split_root_window`],
//! which is a pure function with its own tests.
//!
//! ## Why they are dark, and High risk
//!
//! They cannot make a bound unsound: a deadline only SCHEDULES work, every
//! tightening pass is shrink-only and fails closed on expiry, and the reserve
//! is carved out of the root evaluator's own window rather than minted on top
//! of it (the instance ledger deadline is untouched, so the post-BaB PGD
//! reservation is unaffected). But they are `MoatRisk::High` all the same,
//! because they move budget between phases that each produce bounds, and a
//! bound that is not computed is a verdict that is not reached. Absent (or 0)
//! reproduces the shipped ladder exactly.
//!
//! `NY_ROOT_ALPHA_CAP_SECS`, if set, REPLACES the preset alpha cap inside
//! `shared/init.rs` (that is its documented contract), which means it also
//! replaces the ceiling [`ROOT_ALPHA_FRAC`] min-composes onto it. Do not set
//! both in one A/B.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const ROOT_ARBITRATION: Scope = Scope {
    package: "ny-propagate",
    subsystem: "multi-objective-root-arbitration",
};

const BAB_RESERVE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: ROOT_ARBITRATION,
    role: "arm the split and reserve BaB's share off the top of the root window",
    site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/root.rs:root_bab_window_split",
}];

const ROOT_SPEC_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: ROOT_ARBITRATION,
        role: "reserve the root objective pass's share, so the tightening ladder stops before it",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/root.rs:root_bab_window_split",
    },
    ReaderSite {
        scope: ROOT_ARBITRATION,
        // Not a second env read: the resolved share is THREADED to the pass as
        // the `root_spec_reserve` argument, so only a caller that actually
        // carved the BaB reservation can rebase that grace.
        role: "rebase the root objective grace onto the reserved share instead of the 3 s pinned to an expired bootstrap cap",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/root.rs:compute_root_objective_bounds(root_spec_reserve)",
    },
];

const ROOT_ALPHA_FRAC_READERS: &[ReaderSite] = &[ReaderSite {
    scope: ROOT_ARBITRATION,
    role: "min-compose a share-of-window ceiling onto the bootstrap's fixed root_alpha_cap_secs",
    site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/root.rs:evaluate_root_borrowed",
}];

declare_levers! {
    registry BAB_BUDGET_LEVERS;

    /// `NY_BAB_RESERVE_FRAC` — BaB's guaranteed share of the root window.
    pub BAB_RESERVE_FRAC = LeverDecl {
        name: "NY_BAB_RESERVE_FRAC",
        // Closed, because 0.0 is a meaningful setting: it is the explicit
        // kill switch that restores the un-reserved ladder while leaving the
        // other two knobs parseable, and 1.0 states "root tightening is worth
        // nothing here" without needing a separate token.
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: 1.0 },
        // Unset, not F64(0.0): absent means the arbitration does not exist,
        // and the receipt should say so rather than record a share that was
        // never subtracted.
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Fraction of the multi-objective root window reserved for branch-and-bound and \
subtracted BEFORE any root phase sizes itself. Absent, malformed, or 0.0 leaves \
the shipped ladder byte-identical: BaB stays the residue.

This is the arming gate for the whole `#bab-floor` split — `NY_ROOT_SPEC_FRAC` \
and `NY_ROOT_ALPHA_FRAC` are read only when this one resolves above 0. The \
three shares are clamped and, if they oversubscribe the window, scaled down in \
proportion, so the sweeps' share is never negative.

MEASURED CONTEXT, NOT A MEASURED EFFECT: on cifar100_2024 idx_2176 at 100 s the \
root ladder consumed ~63.5 s of a 72.892 s window and BaB was never entered. \
No armed-vs-unarmed verdict comparison has been retained; this reserves time, \
it does not convert a row.",
        provenance: Provenance::Unmeasured {
            why_ok: "absent or 0.0 subtracts nothing and reproduces the shipped ladder exactly; \
                     deadlines only schedule work, every root tightening pass is shrink-only and \
                     fails closed on expiry, and the reserve is carved out of the root \
                     evaluator's own window rather than added to the instance ledger",
        },
        owner: ROOT_ARBITRATION,
        readers: BAB_RESERVE_READERS,
    };

    /// `NY_ROOT_SPEC_FRAC` — the root objective pass's share of the window.
    pub ROOT_SPEC_FRAC = LeverDecl {
        name: "NY_ROOT_SPEC_FRAC",
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: 1.0 },
        // A real shipped VALUE rather than Unset: once the split is armed, the
        // spec pass must get a share or the reservation just relocates the
        // starvation from BaB to the pass that produces the bounds BaB needs.
        default: DefaultSpec::F64(0.15),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Fraction of the multi-objective root window reserved for `compute_root_objective_bounds`, \
subtracted after `NY_BAB_RESERVE_FRAC` and before the tightening ladder. Read \
only when `NY_BAB_RESERVE_FRAC` is armed.

It has a second, coupled effect: while the split is armed the root objective \
grace is rebased onto this reservation instead of the 3 s `ROOT_SPEC_GRACE` \
pinned to the ALREADY-EXPIRED bootstrap alpha cap. That 3 s is the grace the \
pass blew on idx_2176, and its handler returns terminally, so the pass must own \
a share for the BaB reserve behind it to be reachable at all.",
        provenance: Provenance::Unmeasured {
            why_ok: "read only under an armed NY_BAB_RESERVE_FRAC, so the shipped path never \
                     observes it; the 99-row spec CROWN backward cost on this graph is not \
                     established, which is precisely what an A/B of this knob would price",
        },
        owner: ROOT_ARBITRATION,
        readers: ROOT_SPEC_READERS,
    };

    /// `NY_ROOT_ALPHA_FRAC` — the bootstrap ascent's share of the window.
    pub ROOT_ALPHA_FRAC = LeverDecl {
        name: "NY_ROOT_ALPHA_FRAC",
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: 1.0 },
        default: DefaultSpec::F64(0.30),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Fraction of the multi-objective root window the root-alpha bootstrap ascent may \
consume, min-composed onto the preset's `root_alpha_cap_secs`. Read only when \
`NY_BAB_RESERVE_FRAC` is armed.

It exists because the bootstrap is the one ladder claimant that does NOT scale \
with the remaining budget: `root_alpha_cap_secs: 40` is 51% / 17% / 4% of the \
BaB slice at 100 s / 330 s / 1200 s (see `ny_core::phase_window`). Reserving \
time downstream without also converting that wall into a share would leave the \
sweeps, not the bootstrap, paying for the whole reservation.

The ceiling is applied through the SAME local-phase-cap seam the preset cap \
uses, so an expired ascent still publishes its phase-cap CHECKPOINT; tightening \
the bootstrap's DEADLINE instead would clear `local_phase_cap_applied` and turn \
that same expiry into a hard DeadlineExceeded. An explicit \
`NY_ROOT_ALPHA_CAP_SECS` replaces the preset cap inside `shared/init.rs` and \
therefore also replaces this ceiling.",
        provenance: Provenance::Unmeasured {
            why_ok: "read only under an armed NY_BAB_RESERVE_FRAC; it can only LOWER the \
                     configured cap, and a lowered root-alpha cap degrades tightness rather \
                     than soundness — the checkpoint publishes the sound pre-loop reference \
                     bounds",
        },
        owner: ROOT_ARBITRATION,
        readers: ROOT_ALPHA_FRAC_READERS,
    };
}
