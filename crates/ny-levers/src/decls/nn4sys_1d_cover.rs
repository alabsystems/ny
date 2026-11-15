// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The nn4sys exact one-axis complete-cover prepass (#nn4sys-1d-cover).
//!
//! Three levers, promoted together ON PURPOSE. The master alone is the
//! `alpha_iteration: 1` / `cols=8 -> cols=1` shape: it would deliver
//! `DEFAULT_DEPTH = 8` into an ALL-OR-NOTHING pass (one undecided eligible
//! leaf, cap, or deadline discards the entire prepass), so a promotion that
//! shipped the master by itself would be reachable and inert. The depth key
//! makes the pass's own budget expressible; the slice key fences the wall
//! clock it may take out of the screen that actually decides these rows.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const NN4SYS_1D_COVER_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "nn4sys-scalar-cover",
};

const NN4SYS_1D_COVER_READERS: &[ReaderSite] = &[ReaderSite {
    scope: NN4SYS_1D_COVER_SCOPE,
    role: "the single gate deciding whether the exact one-axis f64 complete \
           cover runs as a prepass in front of the box-refinement clause \
           screen",
    site: "crates/ny-cli/src/commands/beta_crown/verify/nn4sys_scalar_cover.rs \
           (enabled)",
}];

const NN4SYS_1D_COVER_DEPTH_READERS: &[ReaderSite] = &[ReaderSite {
    scope: NN4SYS_1D_COVER_SCOPE,
    role: "the per-axis bisection depth the cover is allowed to reach before \
           it declares an undecided leaf and discards the whole prepass",
    site: "crates/ny-cli/src/commands/beta_crown/verify/nn4sys_scalar_cover.rs \
           (ScalarCoverLimits::from_env)",
}];

const NN4SYS_1D_COVER_SLICE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: NN4SYS_1D_COVER_SCOPE,
    role: "the fraction of the caller's REMAINING deadline the cover may \
           consume; also selects the bounded partial-publication mode, so the \
           prepass can never take the whole screen slice and return nothing",
    site: "crates/ny-cli/src/commands/beta_crown/verify/nn4sys_scalar_cover.rs \
           (partial_cover_deadline)",
}];

declare_levers! {
    registry NN4SYS_1D_COVER_LEVERS;

    /// `NY_NN4SYS_1D_COMPLETE_COVER` — exact-`"1"` master gate for the nn4sys
    /// one-axis exact complete cover. Ships DARK.
    pub NN4SYS_1D_COMPLETE_COVER = LeverDecl {
        name: "NY_NN4SYS_1D_COMPLETE_COVER",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the exact one-axis f64 complete cover as a prepass in front of the \
production box-refinement clause screen. A census of the shipped nn4sys \
properties (reports/nn4sys_exact_1d_census_2026-07-26.csv) finds 97.2-97.8% \
of non-dual clauses and 75.8-79.1% of DUAL clauses reducible to one ranged \
axis, with zero multiplicative-nonlinear and zero dynamic-divisor \
obstructions, so this is the only mechanism on the front that reduces the \
root box COUNT rather than the per-box constant.

Soundness posture (nn4sys is a blind category — a wrong `unsat` is \
uncontradictable, hence High): the cover PUBLISHES only obligations it \
discharged itself, through the same certified outward-rounded f64 machinery \
the screen uses, and its pass is all-or-nothing — one undecided eligible \
leaf, propagation error, unsupported schema, cap, or deadline discards the \
ENTIRE prepass and the unchanged screen decides every clause. It seeds a \
monotone union of `true` flags into the screen; a `false` seed always means \
unproven and proceeds unchanged. Exact `1` arms; with the gate unset the \
prepass never runs and the screen is byte-identical.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark and all-or-nothing: unset is byte-identical to \
                     the historical screen-only path, and an armed run can only \
                     ADD certified per-clause discharges that the screen would \
                     otherwise have to re-derive. No armed-vs-unarmed A/B has run \
                     at official budgets",
        },
        owner: NN4SYS_1D_COVER_SCOPE,
        readers: NN4SYS_1D_COVER_READERS,
    };

    /// `NY_NN4SYS_1D_COVER_DEPTH` — per-axis bisection depth for the cover.
    /// Default 8, which is the reader's own `DEFAULT_DEPTH`; the reader keeps
    /// clamping to its `HARD_MAX_DEPTH` of 24.
    pub NN4SYS_1D_COVER_DEPTH = LeverDecl {
        name: "NY_NN4SYS_1D_COVER_DEPTH",
        kind: LeverKind::U64,
        default: DefaultSpec::U64(8),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Per-axis bisection depth for the one-axis complete cover. NOT optional \
alongside the master: the cover's pass is all-or-nothing, so a depth too \
small for a property does not degrade the result, it DISCARDS the prepass \
and delivers a reachable, correct, inert mechanism — the exact \
legal-value-makes-it-a-no-op shape the preset-arrival ratchet exists to \
catch. The reader retains its own `HARD_MAX_DEPTH = 24` clamp and its \
`centered_start_depth <= depth` validity rule, so an out-of-range value \
falls back to the shipped default rather than widening any bound.

Soundness: depth changes only HOW FAR the cover subdivides before giving \
up. Every leaf bound is the unchanged certified f64 walk, and an unfinished \
cover publishes nothing.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while the master is dark, and a budget knob when it is \
                     not: it can only change whether the cover completes, never \
                     what a completed cover claims",
        },
        owner: NN4SYS_1D_COVER_SCOPE,
        readers: NN4SYS_1D_COVER_DEPTH_READERS,
    };

    /// `NY_NN4SYS_1D_COVER_SLICE` — fraction of the caller's REMAINING
    /// deadline the cover may consume. Default 0.25, admissible [0.02, 0.5].
    pub NN4SYS_1D_COVER_SLICE = LeverDecl {
        name: "NY_NN4SYS_1D_COVER_SLICE",
        kind: LeverKind::F64ClosedTrimmed { min: 0.02, max: 0.5 },
        default: DefaultSpec::F64(0.25),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Fences the cover against the budget it is spending. When the master is armed \
and the caller carries a deadline, this selects the bounded PARTIAL mode and \
gives the cover this fraction of the time REMAINING at entry; the existing \
`caller.min(local)` rule still applies, so the cover can never outlive its \
caller.

A FRACTION, deliberately, not the historical wall-clock \
`NY_NN4SYS_1D_PARTIAL_BUDGET_MS`: nn4sys budgets span 20s to 800s in one \
benchmark, so a millisecond constant is not scale-free, and the unfenced \
worst case is the cover eating the whole 0.95 screen slice and returning \
`None`. That is the same blast radius as this benchmark's own banked \
`sat_escape_branch` A/B, which capped the precheck slice and took 12 dual \
rows to 1.

This declaration ships a DEFAULT, not `Unset`, so the fence is in force the moment the master is armed. Say so plainly: that CHANGES the historical env-only recipe `NY_NN4SYS_1D_COMPLETE_COVER=1`, which ran an unbounded cover against the caller's whole deadline. Nothing shipped depended on it — the master was dark and env-only, i.e. unreachable in competition — but a null measured on the new arm must not be read as a regression against the old one.

Soundness: a budget only. The cover publishes a clause only after that clause's entire group cover retires; every unfinished clause stays false for the screen and the downstream verifier.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while the master is dark. Armed, it can only take proof \
                     time away from the cover and give it back to the screen, \
                     never the reverse, and it cannot change what either lane \
                     certifies",
        },
        owner: NN4SYS_1D_COVER_SCOPE,
        readers: NN4SYS_1D_COVER_SLICE_READERS,
    };
}
