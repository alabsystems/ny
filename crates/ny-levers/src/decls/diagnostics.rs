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

const FORCE_SELFCHECK_FAIL_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BAB_BOUND_AUTHORITY_SCOPE,
    role: "force the BaB-bound authority self-check to fail, so the refusal path can be exercised on hardware that would otherwise pass it",
    site: "crates/ny-gpu/src/wgpu_device/ops/bab_bound_authority.rs:env_forces_selfcheck_failure",
}];

declare_levers! {
    registry DIAGNOSTIC_LEVERS;

    /// `NY_PATCHES_FINITE_EXPIRY` — decide finite Patches authority by expiry, not presence.
    pub PATCHES_FINITE_EXPIRY = LeverDecl {
        name: "NY_PATCHES_FINITE_EXPIRY",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Ships OFF, and OFF is byte-identical to the shipped path.

WHAT IT CHANGES. Under hard finite authority the native Patches routes are
refused whenever a deadline is merely PRESENT. Since every scored run carries
one, that refusal fires on every conv row — and it is a DEAD END rather than a
fallback: the Dense carrier it produces goes to
`dispatch_backward_layer_finite_boundary`, which declines every layer family
except SkipMerge/ReLU/Where/Div. The node therefore ends with reference bounds
and no CROWN at all, neither structured nor dense. Armed, this lever decides the
same refusal by deadline EXPIRY, so a live deadline keeps the native route and
an expired one still refuses.

WHY IT IS NOT THE DEFAULT, stated as a measurement rather than a worry. On the
20-row `relusplitter` biasfield subset it converts NOTHING: 3 sat / 17 timeout
in both arms, identical row by row. The blocker it removes is real and
measurable in the logs — the `has no fully cooperative finite-deadline dispatch
route` declines disappear and tight work runs — but on `cifar_bias_field_46` the
run then exhausts its budget inside `FaerCpuGemmEngine::gemm_f64_with_deadline`
instead. That row proved in 35.7 s at `97fb4bd6a` via `requested=wgpu
source=preset`, so what is still missing is GPU ROUTING for that backward, which
belongs to the wgpu verdict-authority program. Promoting a default on evidence
that converts zero rows is exactly what the moat rule forbids.

MoatRisk::High because the arm gives up an interruptibility invariant: the
native Patches kernels poll their dominant contraction but own unreceipted
allocation and scanning phases, so an armed run can overrun by a bounded single
layer step. No bound is at risk in either arm — both routes are sound, and the
refused path was losing PRECISION, not gaining safety.",
        provenance: Provenance::Unmeasured {
            why_ok: "ships off and byte-identical; the armed arm is measured \
                     verdict-neutral on 20 biasfield rows, which is evidence that it is \
                     safe to try and NOT evidence that it should ship",
        },
        owner: PATCHES_FINITE_SCOPE,
        readers: PATCHES_FINITE_EXPIRY_READERS,
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
