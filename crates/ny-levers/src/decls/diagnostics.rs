// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Legacy convolution-Patches diagnostics.
//!
//! The other late propagation controls live in [`super::dark_probes`]. This
//! module retains only the distinct presence-style Patches diagnostic, avoiding
//! duplicate declarations while keeping its historical parser enumerable.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const PATCHES_DIAGNOSTIC_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "graph-crown-patches-diagnostics",
};

const BAB_BOUND_AUTHORITY_SCOPE: Scope = Scope {
    package: "ny-gpu",
    subsystem: "bab-bound-authority-selfcheck",
};

const PATCHES_FINITE_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "patches-finite-authority",
};

const PATCHES_FINITE_EXPIRY_READERS: &[ReaderSite] = &[ReaderSite {
    scope: PATCHES_FINITE_SCOPE,
    role: "decide hard finite authority over the native Patches routes by deadline EXPIRY instead of deadline PRESENCE",
    site: "crates/ny-propagate/src/network/core/sequential/crown/patches_step.rs:hard_finite_authority_refuses_patches",
}];

const CROWN_PARTIAL_FINITE_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "crown-partial-finite-authority",
};

/// ONE latch, TWO decision sites. `crown_partial_expiry_armed` is the only
/// place the name is read; both root-cause-D set-mates in that file — the
/// GPU-resident partial backward route (set-mate 1/2) and the sparse-Patches
/// seed discovery (set-mate 2/2) — call it and feed the same pure predicate
/// `finite_authority_declines_partial`. Listing the latch rather than the two
/// call sites keeps this list a truthful inventory of reads.
const CROWN_PARTIAL_FINITE_EXPIRY_READERS: &[ReaderSite] = &[ReaderSite {
    scope: CROWN_PARTIAL_FINITE_SCOPE,
    role: "latch, once per process, whether the CROWN-IBP partial GPU route and sparse-seed discovery decide by deadline EXPIRY instead of deadline PRESENCE",
    site: "crates/ny-propagate/src/network/ibp/crown_partial.rs:crown_partial_expiry_armed",
}];

const FORCE_SELFCHECK_FAIL_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BAB_BOUND_AUTHORITY_SCOPE,
    role: "force the BaB-bound authority self-check to fail, so the refusal path can be exercised on hardware that would otherwise pass it",
    site: "crates/ny-gpu/src/wgpu_device/ops/bab_bound_authority.rs:env_forces_selfcheck_failure",
}];

declare_levers! {
    registry DIAGNOSTIC_LEVERS;

    /// `NY_PATCHES_FINITE_EXPIRY` — decide finite Patches authority by expiry, not
    /// presence. Ships ARMED; `=0` is the kill switch back to the presence test.
    pub PATCHES_FINITE_EXPIRY = LeverDecl {
        name: "NY_PATCHES_FINITE_EXPIRY",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "\
Ships ON: absent means ARMED. `NY_PATCHES_FINITE_EXPIRY=0` is the kill switch and
restores the old presence behaviour exactly; exact `1` arms it explicitly, and
any other token is a recorded rejection that leaves the armed default in place.

WHAT IT CHANGES. Under hard finite authority the native Patches routes were
refused whenever a deadline was merely PRESENT. Since every scored run carries
one, that refusal fired on every conv row — and it is a DEAD END rather than a
fallback: the Dense carrier it produces goes to
`dispatch_backward_layer_finite_boundary`, which declines every layer family
except SkipMerge/ReLU/Where/Div. The node therefore ended with reference bounds
and no CROWN at all, neither structured nor dense. This lever decides the same
refusal by deadline EXPIRY, so a live deadline keeps the native route and an
expired one still refuses; disarming it with `0` puts the presence test back.

WHY IT SHIPS ARMED, stated as measurements rather than as a preference.
NON-REGRESSION FIRST: on the pinned 20-row `relusplitter` biasfield subset the
two arms are 3 sat / 17 timeout, IDENTICAL ROW BY ROW, so the flip converts
nothing there and loses nothing there. COST: on `cifar_bias_field_46` the
`graph-bab-bootstrap` phase goes 37.3 s -> 1.4 s armed, because the conv below a
ReLU keeps a PATCHES carrier instead of densifying. QUALITY: at `8c393486c`,
armed and with the ordering fix, that row PROVES unsat at a 300 s budget with
bounds marginally TIGHTER than the last-good `97fb4bd6a` — alpha iter-0
lower_sum -167.82 vs -168.46, layer-4 width_sum 726.70 vs 726.89.

WHAT IS STILL OPEN, so the promotion is not oversold. `55ec3d0bf` quantifies the
residual as 2.36x walk cost (collection 63.3 s vs 26.8 s), which is why the row
does not yet fit its 60 s official budget. That cost lives in the cooperative
finite routes' serial phases and the scalar `incoming_error_product` loop, not
in this gate: disarming the lever does not recover it, it only restores the
dead-end refusal.

MoatRisk::High because the armed arm gives up an interruptibility invariant
WITHIN a layer step: the native Patches kernels poll their dominant contraction
but own unreceipted allocation and scanning phases, so an armed run can overrun
by a bounded single layer step. No bound is at risk in either arm — both routes
are sound, and the refused path was losing PRECISION, not gaining safety. The
kill switch exists for exactly that overrun: a caller that cannot tolerate a
one-step overshoot sets `0` and takes the old refusal back.",
        provenance: Provenance::Measured {
            commit: "8c393486c",
            date: "2026-08-17",
            artifact: "docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md, sections 'THE \
                       DENSIFYING SITE, FOUND AND FIXED (2026-08-17)' (bootstrap and \
                       the 20-row subset) and 'QUALITY CHAIN CLOSED (2026-08-17)' (the \
                       300 s proof); commit messages 8c393486c and 55ec3d0bf",
            delta: "cifar_bias_field_46, re-measured 2026-08-18 on the CURRENT tree: \
                    `graph-bab-bootstrap` 0.9 s disarmed vs 36.9 s armed. That \
                    inversion is deliberate and is the point — the disarmed arm is \
                    cheap because it DECLINES to reference bounds, the armed arm pays \
                    for the real structured walk. (An earlier 37.3 -> 1.4 s reading \
                    predates `b357b9de9`, which made the hard-authority route take a \
                    deadline-plumbed Dense retry instead of a typed refusal; do not \
                    cite it.) At 8c393486c, armed plus the ordering fix, that row \
                    PROVES unsat at a 300 s budget with bounds marginally TIGHTER than \
                    the last-good 97fb4bd6a: alpha iter-0 lower_sum -167.82 vs -168.46, \
                    layer-4 width_sum 726.70 vs 726.89. Verdict-neutral where it does \
                    not win: the pinned 20-row relusplitter biasfield subset is 3 sat / \
                    17 timeout in BOTH arms, identical row by row, RE-RUN on this tree \
                    2026-08-18 with the arms explicit — zero conversions AND \
                    zero regressions. Residual, unfixed by this gate and quantified at \
                    55ec3d0bf: 2.36x walk cost (collection 63.3 s vs 26.8 s), so the row \
                    still does not fit its 60 s official budget.",
        },
        owner: PATCHES_FINITE_SCOPE,
        readers: PATCHES_FINITE_EXPIRY_READERS,
    };

    /// `NY_CROWN_PARTIAL_FINITE_EXPIRY` — the CROWN-IBP partial sibling of
    /// [`PATCHES_FINITE_EXPIRY`]. Ships DARK; exact `1` is the only arming token.
    pub CROWN_PARTIAL_FINITE_EXPIRY = LeverDecl {
        name: "NY_CROWN_PARTIAL_FINITE_EXPIRY",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Ships OFF: absent means the historical deadline-PRESENCE guards stay in place, \
byte for byte. Exact `1` arms; exact `0` is an explicit, recorded disarm; every \
other token (`true`, `yes`, `01`, ` 1`, the empty string) is a recorded \
rejection that leaves the dark default. Declared `Bool` rather than `Presence` \
because the reader it replaced compared the raw `OsStr` against exactly `\"1\"`, \
so `NY_CROWN_PARTIAL_FINITE_EXPIRY=0` genuinely means DISARMED and a non-UTF-8 \
value never armed it either.

WHAT IT CHANGES. Two sites in `network/ibp/crown_partial.rs` declined whenever a \
deadline was merely PRESENT: the GPU-resident partial backward route \
(set-mate 1/2) and the sparse-Patches seed discovery (set-mate 2/2). Every \
scored run carries a deadline, so the GPU route ran only in tests, and every \
scored run was forced onto the FULL virtual-identity seed — the strictly larger \
allocation. Armed, both decide by EXPIRY instead: a live unexpired deadline \
keeps the native route and the sparse seed, an expired one still declines. \
Arming never extends a deadline — the walk's pre-existing entry/exit expiry \
checks (`publish_concretized_crown`, `check_partial_deadline`, and the GPU \
pre/post-dispatch checks) still refuse any late publication.

WHY IT IS THE WHOLE SET OR NOTHING. These are the last two unswitched members of \
BUG #18 root cause D; their mates in `sequential/crown/backward_step.rs` \
(`finite_authority_refuses_per_layer_ibp`) already decide by expiry \
unconditionally. The audit's standing finding is that set-mates switch together \
and partial fixes measure exactly zero — the same decline reappears one node \
later — so arming this lever is the only configuration in which the D set can \
measure as nonzero at all.

WHY IT IS DARK. Its introducing commit benchmarked nothing and claims no \
conversion (`docs/DEADLINE_PRESENCE_FIX_2026-08-19.md`, 'What is deliberately \
NOT claimed'). The A/B is pre-registered there — baseline / armed / \
deadline-absent oracle at the official per-instance budget — and it has an \
ENGAGEMENT GATE that must be read before any verdict: the armed arm must show \
`[deadline-preserve] crown-partial-salvage: saved>0`. A null with those \
counters at zero is a WRONG-LANE result, not a negative.

MoatRisk::High, and not because either arm is unsound. Neither site creates, \
loosens or tightens a bound value: both are route/seed SELECTION, both arms \
produce sound enclosures, and rows the sparse seed does not track merge from the \
same sound IBP bounds (`merge_sparse_crown_with_ibp`), so seed choice cannot \
manufacture a value. It is High for two concrete reasons. First, changing which \
route computes a verdict-relevant INTERMEDIATE CROWN bound changes the \
intermediate bounds that get published, and those feed the verdict. Second, the \
armed arm gives up an interruptibility invariant WITHIN the walk: the host layer \
extraction inside `try_gpu_crown_partial_backward` is not internally polled, so \
an armed run can overrun by that one bounded preparation phase — the same \
accepted tradeoff class as `NY_PATCHES_FINITE_EXPIRY`. Disarming with `0` (or \
leaving it absent) takes the historical decline back.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark and byte-identical unarmed; the introducing commit \
                     executed nothing and pre-registers the A/B instead of claiming it, \
                     so there is no measurement to cite and no promotion being made. \
                     Unlike its sibling NY_PATCHES_FINITE_EXPIRY, these two sites have \
                     no armed-vs-unarmed scored evidence at all",
        },
        owner: CROWN_PARTIAL_FINITE_SCOPE,
        readers: CROWN_PARTIAL_FINITE_EXPIRY_READERS,
    };

    /// `NY_FORCE_GPU_BAB_BOUND_SELFCHECK_FAIL` — force the authority self-check to refuse.
    pub FORCE_GPU_BAB_BOUND_SELFCHECK_FAIL = LeverDecl {
        name: "NY_FORCE_GPU_BAB_BOUND_SELFCHECK_FAIL",
        kind: LeverKind::Presence,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::None,
        doc: "\
Forces the GPU BaB-bound authority self-check to FAIL. Set to anything — the \
reader is `var_os(..).is_some()`, latched once in a `OnceLock` so the answer \
cannot change mid-process.

IT ONLY EVER REMOVES AUTHORITY. There is no arm of this lever that grants a \
verdict, admits a bound, or relaxes a check; failing the self-check sends the \
caller to its existing sound fallback. That is why it is `MoatRisk::None` \
despite touching a verdict-authority path, and it is also why it is worth \
having: hardware that passes the self-check cannot otherwise exercise the \
refusal branch, and an untested refusal path is how a fail-closed design \
quietly stops being one.

Declared as `Presence` because that is what the reader does. Rounding it to \
`Bool` would report `false` in the receipt for a run started with \
`NY_FORCE_GPU_BAB_BOUND_SELFCHECK_FAIL=0` — a run whose self-check was in fact \
forced to fail.",
        provenance: Provenance::Unmeasured {
            why_ok: "unset by default and strictly authority-removing; every arm is at \
                     least as conservative as the default, so there is no promotion to justify",
        },
        owner: BAB_BOUND_AUTHORITY_SCOPE,
        readers: FORCE_SELFCHECK_FAIL_READERS,
    };

    /// `NY_CONV_PATCHES_DEBUG` — legacy nonempty/nonzero Patches diagnostics.
    pub CONV_PATCHES_DEBUG = LeverDecl {
        name: "NY_CONV_PATCHES_DEBUG",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Enables per-node convolution-Patches routing diagnostics for every present, \
nonempty value except exact `0`. This intentionally preserves the older \
presence-style parser rather than narrowing it to an exact-one Boolean during \
migration. Absence and the two explicit off spellings (`0` and the empty \
string) are dark. Output is observational but can perturb deadline timing.",
        provenance: Provenance::Unmeasured {
            why_ok: "legacy diagnostic remains dark when absent; armed-vs-unarmed deadline and verdict parity has not been measured",
        },
        owner: PATCHES_DIAGNOSTIC_SCOPE,
        readers: &[
            ReaderSite {
                scope: PATCHES_DIAGNOSTIC_SCOPE,
                role: "emit Conv2d Patches backward diagnostics",
                site: "crates/ny-propagate/src/layers/convolution/conv2d/bound_patches.rs",
            },
            ReaderSite {
                scope: PATCHES_DIAGNOSTIC_SCOPE,
                role: "emit explicit graph-alpha Patches diagnostics",
                site: "crates/ny-propagate/src/network/graph_alpha/bounds/alpha_explicit.rs",
            },
            ReaderSite {
                scope: PATCHES_DIAGNOSTIC_SCOPE,
                role: "emit graph-alpha tightening Patches diagnostics",
                site: "crates/ny-propagate/src/network/graph_alpha/bounds/crown_tighten.rs",
            },
            ReaderSite {
                scope: PATCHES_DIAGNOSTIC_SCOPE,
                role: "emit plain Graph-CROWN Patches fallback diagnostics",
                site: "crates/ny-propagate/src/network/graph_crown/propagation.rs:dispatch_plain_patches_or_fallback",
            },
        ],
    };

    /// `NY_DUMP_NODE_BOUNDS` — per-layer CROWN-IBP bound summary at publication.
    pub DUMP_NODE_BOUNDS = LeverDecl {
        name: "NY_DUMP_NODE_BOUNDS",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Prints a per-layer min/max/total-width summary of the CROWN-IBP bounds at the \
point they are published, for hunting divergence between two binaries on the \
same row. Exact `1` arms it; absence and every other value leave it dark, \
matching the reader's `== Some(\"1\")` test verbatim.

Print-only: it reads the published bounds and writes to stderr, feeding no \
value, lifetime or ordering that any bound or deadline comparison depends on. \
MoatRisk::Low rather than None only because the formatting and stderr traffic \
cost real time on a deadline-sensitive row, which is the same reason every other \
diagnostic in this module carries Low. Its own comment calls it TEMPORARY; \
delete the lever with the diagnostic.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one diagnostic that publishes nothing and changes \
                     no bound; the dark arm is the shipped path and is unaffected",
        },
        owner: PATCHES_DIAGNOSTIC_SCOPE,
        readers: &[
            ReaderSite {
                scope: PATCHES_DIAGNOSTIC_SCOPE,
                role: "dump per-layer CROWN-IBP bounds at publication for binary-vs-binary divergence hunting",
                site: "crates/ny-propagate/src/network/ibp/crown_ibp.rs (publication)",
            },
            ReaderSite {
                scope: PATCHES_DIAGNOSTIC_SCOPE,
                role: "dump the same summary at the per-node site b357b9de9 added",
                site: "crates/ny-propagate/src/network/ibp/crown_ibp.rs (per-node)",
            },
        ],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read_with;

    /// The partial sibling ships DARK and is armed by exactly one token. The
    /// reader it replaced compared an `OsStr` against `"1"`, so `"0"` is a real
    /// disarm rather than a near-miss and nothing else may arm it.
    #[test]
    fn crown_partial_expiry_is_dark_and_armed_only_by_exact_one() {
        let armed = |raw: Option<&str>| {
            read_with(&CROWN_PARTIAL_FINITE_EXPIRY, |_| raw.map(str::to_owned))
                .value
                .as_bool()
        };
        assert!(armed(Some("1")));
        for token in [
            None,
            Some("0"),
            Some("true"),
            Some(""),
            Some(" 1"),
            Some("2"),
        ] {
            assert!(
                !armed(token),
                "{token:?} must not arm the partial expiry lever"
            );
        }
        assert_eq!(CROWN_PARTIAL_FINITE_EXPIRY.bucket, Bucket::Debug);
        assert!(matches!(
            CROWN_PARTIAL_FINITE_EXPIRY.provenance,
            Provenance::Unmeasured { .. }
        ));
    }

    #[test]
    fn patches_debug_preserves_legacy_nonempty_nonzero_parser() {
        let enabled = |raw: Option<&str>| {
            read_with(&CONV_PATCHES_DEBUG, |_| raw.map(str::to_owned))
                .value
                .as_str()
                .is_some_and(|value| !value.is_empty() && value != "0")
        };
        assert!(!enabled(None));
        assert!(!enabled(Some("")));
        assert!(!enabled(Some("0")));
        assert!(enabled(Some("1")));
        assert!(enabled(Some("true")));
        assert!(enabled(Some(" 0")));
    }
}
