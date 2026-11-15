// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The forward-linear reference gates and the f32 value-GEMM seam.
//!
//! Four historical `NY_*` switches steer the certified forward-linear (FL)
//! reference surface in `ny-propagate`: two kill switches whose absence leaves
//! the reference ENGAGED, one default-dark research gate, and the
//! precision-negative f32 seam. None of them had a declaration, so every
//! executable spelling of the four names — the raw readers AND every test that
//! pins the ambient environment around an FL fixture — was undeclared literal
//! debt on the source frontier (`crates/ny-levers/tests/source_frontier.rs`),
//! and a new FL test could only scrub them by adding more of it
//! (#fl-alpha-composition: `fl_composition_tests` re-landed with its scrub
//! folded into ny-propagate's shared `tests::with_default_fl_reference_env`,
//! which still spelled the four names as new legacy rows).
//!
//! Declaring them is behaviour-preserving: [`LeverKind::Bool`] is the
//! historical exact-`"1"` predicate on every input (the tests below pin the
//! declarations to it), and every live reader now reads THROUGH its
//! declaration (`ny_levers::read`), so the four raw `std::env::var` reads left
//! the reader frontier and a reader can no longer drift from the receipt
//! (ny-propagate's `live_fl_reference_gates_agree_with_their_declarations` and
//! `f32_seam_env_fallback_agrees_with_its_declaration` pin the live gates
//! against these declarations). What the declaration buys is a typed handle
//! (`ny_levers::decls::forward_linear::*.name`) for tests and tools, a receipt
//! entry, and a recorded search-space decision (`space.rs` `UNSAFE_AXES`).
//! The two engaged-by-absence kill switches carry the
//! `LEGACY-ARMED-UNQUALIFIED` marker: their engaged policies ship without a
//! retained discriminating current-path verdict A/B.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const FL_REFERENCE_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "forward-linear-reference",
};

const ALPHA_REFERENCE_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "alpha-reference-collection",
};

const SPEC_PROPAGATION_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "graph-crown-spec-propagation",
};

const INPUT_SPLIT_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "input-split-bounds-eval",
};

const FL_F32_SEAM_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "forward-linear-f32-seam",
};

const CLI_DISJUNCTIVE_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "disjunctive-attack-cache-warmer",
};

const CLI_BETA_CROWN_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "beta-crown-preset-bridge",
};

const NO_FORWARD_LINEAR_REF_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: FL_REFERENCE_SCOPE,
        role: "the single read, through this declaration: exact `1` withdraws the shared \
               conv-DAG forward-linear reference source for every consumer below",
        site: "crates/ny-propagate/src/network/core/graph/forward_linear.rs:\
               GraphNetwork::forward_linear_reference_enabled",
    },
    ReaderSite {
        scope: FL_REFERENCE_SCOPE,
        role: "refuse Step-1 intermediate-bound collection (and the image-only reference \
               admission built on it) from the cached forward-linear map",
        site: "crates/ny-propagate/src/network/core/graph/forward_linear.rs:\
               GraphNetwork::should_collect_forward_linear_intermediate_reference",
    },
    ReaderSite {
        scope: ALPHA_REFERENCE_SCOPE,
        role: "refuse the forward-linear alpha reference attempt, both on conv DAGs (through \
               the image-reference admission above) and on sequential ConvTranspose / research \
               sequential-Conv2d chains, falling through to the shipped collectors",
        site: "crates/ny-propagate/src/network/graph_alpha/bounds/alpha.rs:\
               collect_alpha_reference_bounds_with_engine_and_source_routed",
    },
    ReaderSite {
        scope: SPEC_PROPAGATION_SCOPE,
        role: "refuse the alpha-optimized forward-linear spec-margin rebuild candidate",
        site: "crates/ny-propagate/src/network/graph_crown/spec_propagation/core.rs:\
               propagate_crown_with_specs_and_engine_with_linear_and_reference_bounds_and_deadline_and_truncation",
    },
    ReaderSite {
        scope: CLI_DISJUNCTIVE_SCOPE,
        role: "skip warming the forward-linear map cache while the disjunctive attack slice \
               is handed to branch-and-bound",
        site: "crates/ny-cli/src/commands/beta_crown/verify/disjunctive.rs:\
               verify_multi_clause_disjunction",
    },
];

const NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: FL_REFERENCE_SCOPE,
        role: "the single read, through this declaration: exact `1` withdraws the \
               ConvTranspose2d/BatchNorm image forward-linear composition (and the sequential \
               cGAN reference arm)",
        site: "crates/ny-propagate/src/network/core/graph/forward_linear.rs:\
               GraphNetwork::forward_linear_conv_transpose_reference_enabled",
    },
    ReaderSite {
        scope: FL_REFERENCE_SCOPE,
        role: "admit the sequential cGAN image surface (Conv2d + ConvTranspose2d, or the \
               graph-local no-transpose arm) to Step-1 forward-linear intermediates",
        site: "crates/ny-propagate/src/network/core/graph/forward_linear.rs:\
               GraphNetwork::should_collect_forward_linear_intermediate_reference",
    },
    ReaderSite {
        scope: FL_REFERENCE_SCOPE,
        role: "select whether the cached forward-linear state may compose ConvTranspose2d, \
               which also keys the cache fingerprint",
        site: "crates/ny-propagate/src/network/core/graph/forward_linear.rs:\
               collect_forward_linear_state_cached_impl",
    },
    ReaderSite {
        scope: ALPHA_REFERENCE_SCOPE,
        role: "admit the ConvTranspose surface to the sequential forward-linear alpha \
               reference attempt",
        site: "crates/ny-propagate/src/network/graph_alpha/bounds/alpha.rs:\
               collect_alpha_reference_bounds_with_engine_and_source_routed",
    },
    ReaderSite {
        scope: INPUT_SPLIT_SCOPE,
        role: "route input-split graph spec-CROWN through the forward-linear reference on \
               ConvTranspose graphs",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/bounds_eval.rs:\
               graph_spec_crown_with_mul_binary_and_truncation",
    },
];

const FORWARD_LINEAR_SEQ_CONV_REF_READERS: &[ReaderSite] = &[ReaderSite {
    scope: ALPHA_REFERENCE_SCOPE,
    role: "the single read, through this declaration: exact `1` admits plain-Conv2d \
           SEQUENTIAL chains to the forward-linear alpha reference attempt \
           (fix_interm_bounds only, fail-closed fallthrough to the shipped collectors)",
    site: "crates/ny-propagate/src/network/graph_alpha/bounds/alpha.rs:\
           collect_alpha_reference_bounds_with_engine_and_source_routed",
}];

const FORWARD_LINEAR_F32_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: FL_F32_SEAM_SCOPE,
        role: "the single live read: the installed (already layered) process-global override \
               first, then exact `1` from the environment through this declaration",
        site: "crates/ny-propagate/src/network/core/graph/forward_linear.rs:\
               forward_linear_f32_seam_armed",
    },
    ReaderSite {
        scope: FL_F32_SEAM_SCOPE,
        role: "choose the f32 or f64 value-GEMM implementation inside the image composition \
               (defers to the reader above)",
        site: "crates/ny-propagate/src/network/core/graph/forward_linear/image.rs:\
               forward_linear_f32_gemm_enabled",
    },
    ReaderSite {
        scope: FL_F32_SEAM_SCOPE,
        role: "key the forward-linear map cache fingerprint and the calibrated build rate \
               on the seam mode",
        site: "crates/ny-propagate/src/network/core/graph/forward_linear.rs:\
               collect_forward_linear_state_cached_impl",
    },
    ReaderSite {
        scope: CLI_BETA_CROWN_SCOPE,
        role: "resolve the typed model.forward_linear_f32_seam key against this declaration \
               (a scored run's frozen LeverSet, which is its flight receipt; interactively the \
               environment over the key) and install the result as the process-global \
               override before any bound is computed",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_forward_linear_f32_seam",
    },
];

declare_levers! {
    registry FORWARD_LINEAR_LEVERS;

    /// `NY_NO_FORWARD_LINEAR_REF` — exact-`"1"` kill switch for the shared
    /// conv-DAG forward-linear reference source.
    pub NO_FORWARD_LINEAR_REF = LeverDecl {
        name: "NY_NO_FORWARD_LINEAR_REF",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact-`\"1\"` kill switch for the certified forward-linear (DeepPoly-style) \
reference source on convolution graphs (#vnncomp-image-forward-linear). Absent \
(the shipped state) leaves the source ENGAGED at every consumer: Step-1 \
intermediate collection, the alpha reference collection, the alpha-optimized \
spec-margin rebuild, and the CLI attack-phase cache warmer all consult ONE \
policy, `GraphNetwork::forward_linear_reference_enabled`. `\"1\"` withdraws it \
everywhere at once and each consumer falls through to its non-forward-linear \
collector; explicit `\"0\"` is an admissible no-op; any other token is a \
recorded rejection that leaves the default in place.

MIGRATION NOTE. The reader was a raw \
`std::env::var(\"NY_NO_FORWARD_LINEAR_REF\").ok().as_deref() == Some(\"1\")`; \
`LeverKind::Bool` gives the same answer on every input (absent, malformed and \
non-UTF-8 all leave the source engaged), so declaring it changed no decision, \
and the reader now reads through this declaration (`ny_levers::read`), so the \
raw read left the reader frontier.

MoatRisk::High: withdrawing the source swaps the intermediate bounds every \
conv-DAG consumer relaxes around, and the two collectors are incomparable, so \
arming it can move a published bound in either direction. \
LEGACY-ARMED-UNQUALIFIED: the engaged reference policy was introduced by \
d6ff31831 (first finite cifar100 root bounds, against a vacuous baseline) with \
a four-verdict regression battery, not by a retained discriminating \
current-path verdict A/B. Phase 0 has no `DefaultStatus` field yet, so this \
exact marker keeps that debt explicit; Bucket::Debug classifies the explicit \
opt-out, not the evidence status of the engaged policy.",
        provenance: Provenance::Unmeasured {
            why_ok: "explicit kill switch around a tracked LEGACY-ARMED-UNQUALIFIED policy: false \
                     is the historical default-ON reference policy, byte-for-byte, and the armed \
                     arm only withdraws a certified reference source so every consumer falls \
                     back to a collector it already ships",
        },
        owner: FL_REFERENCE_SCOPE,
        readers: NO_FORWARD_LINEAR_REF_READERS,
    };

    /// `NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF` — exact-`"1"` kill switch for
    /// the ConvTranspose2d/BatchNorm image forward-linear surface.
    pub NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF = LeverDecl {
        name: "NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact-`\"1\"` kill switch for the certified dense-f64 ConvTranspose2d/BatchNorm \
image forward-linear composition (#cgan-fwdlin-ref). Absent (the shipped state) \
keeps the surface ENGAGED: the forward-linear state may compose ConvTranspose2d, \
the sequential cGAN image graph is admitted to Step-1 forward-linear \
intermediates, and input-split spec-CROWN routes through the reference. `\"1\"` \
withdraws it at every site; the shared `NY_NO_FORWARD_LINEAR_REF` kill switch \
also still applies. Explicit `\"0\"` is an admissible no-op; any other token is \
a recorded rejection that leaves the default in place.

MIGRATION NOTE. The reader was a raw exact-`\"1\"` `std::env::var` comparison, \
which `LeverKind::Bool` reproduces on every input; it now reads through this \
declaration (`ny_levers::read`), so the raw read left the reader frontier.

MoatRisk::High: the reader's own documentation records the engaged surface \
moving the `cGAN_imgSz32_nCh_1 prop_1` root from [-1048.89, 2204.65] to \
[-2.033398, 0.752635] with a 3277-row enclosure audit and byte-inertness on the \
28 non-cGAN 2025 benchmarks; withdrawing it can therefore move cGAN bounds and \
verdicts. LEGACY-ARMED-UNQUALIFIED: those are bound and enclosure records, and \
the same documentation says they are not an audited verdict-parity claim, so \
the engaged surface ships without a retained discriminating current-path \
verdict A/B. Phase 0 has no `DefaultStatus` field yet, so this exact marker \
keeps that debt explicit; Bucket::Debug classifies the explicit opt-out.",
        provenance: Provenance::Unmeasured {
            why_ok: "explicit kill switch around a tracked LEGACY-ARMED-UNQUALIFIED surface: false \
                     is the historical default-ON ConvTranspose reference surface, byte-for-byte, \
                     and the armed arm only withdraws a tighter certified enclosure in favour of \
                     the collectors the surface replaced",
        },
        owner: FL_REFERENCE_SCOPE,
        readers: NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF_READERS,
    };

    /// `NY_FORWARD_LINEAR_SEQ_CONV_REF` — default-dark research gate for
    /// forward-linear alpha references on plain-Conv2d sequential chains.
    pub FORWARD_LINEAR_SEQ_CONV_REF = LeverDecl {
        name: "NY_FORWARD_LINEAR_SEQ_CONV_REF",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact-`\"1\"` research gate (#cora-fwdlin-ref) that lets the alpha reference \
collection try the certified forward-linear map FIRST on SEQUENTIAL chains that \
contain a plain Conv2d (the cora cifar10 set, where the single-threaded deep-seq \
CROWN-IBP sweep was recorded burning ~14 s of a 25 s budget before alpha \
starts). Requires fix_interm_bounds and the shared forward-linear reference \
policy; any fail-closed refusal falls through to the shipped collectors. Absent \
(the shipped state) is byte-identical to the historical routing; explicit \
`\"0\"` is an admissible no-op; any other token is a recorded rejection.

MIGRATION NOTE. The reader was a raw exact-`\"1\"` `std::env::var` comparison, \
which `LeverKind::Bool` reproduces on every input; it now reads through this \
declaration (`ny_levers::read`), so the raw read left the reader frontier.

MoatRisk::High: armed, it changes which intermediate enclosures alpha optimizes \
against on those chains, so it can move a published bound. It has no typed \
preset key and stays a manual research gate.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and byte-identical when absent; the armed arm only \
                     substitutes a certified forward-linear reference and falls through to the \
                     shipped collectors on any refusal",
        },
        owner: ALPHA_REFERENCE_SCOPE,
        readers: FORWARD_LINEAR_SEQ_CONV_REF_READERS,
    };

    /// `NY_FORWARD_LINEAR_F32` — legacy environment route of the sound but
    /// looser f32 value-GEMM seam (typed key: `model.forward_linear_f32_seam`).
    pub FORWARD_LINEAR_F32 = LeverDecl {
        name: "NY_FORWARD_LINEAR_F32",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact-`\"1\"` legacy environment route of the forward-linear f32 value-GEMM seam \
(#fl-f32-seam-key): the composition's big value GEMMs (`A.W`, `|A|.|W|`) run in \
f32 with the Higham `gamma_{K+4}^f32 * S` error plus an FTZ underflow guard \
discharged into the bias through the same certified-error channel as the f64 \
path, while the S-base GEMMs stay f64. SOUND but LOOSER: the seam's own \
documentation records 0 gains and 1 regression on cifar100 \
(docs/CIFAR100_BOUND_PARITY_ORACLE.md), so it ships dark. Absent and explicit \
`\"0\"` keep the shipped f64 path; any other token is a recorded rejection.

DELIVERY. The scored route is the typed `model.forward_linear_f32_seam` preset \
key. `resolve_forward_linear_f32_seam` (ny-cli) layers this variable OVER the \
key exactly as the flight receipt does (a scored run installs its frozen \
`LeverSet` value; an interactive run uses `read_over_config`) and installs the \
result as the process-global override that the live reader \
`forward_linear_f32_seam_armed` consults first, so the run and the receipt \
agree when both layers are set and disagree. Without an installed override \
the reader resolves this declaration (`ny_levers::read`); the raw read left the \
reader frontier. `vnncomp_scripts/run_instance.sh` exports no `NY_*`, so on a \
scored run only the preset layer exists.",
        provenance: Provenance::Unmeasured {
            why_ok: "false keeps the shipped f64 value GEMMs byte-for-byte; the armed arm is a \
                     certified (sound) enclosure that can only be looser, and it is kept dark \
                     on the recorded cifar100 regression",
        },
        owner: FL_F32_SEAM_SCOPE,
        readers: FORWARD_LINEAR_F32_READERS,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_with, Source};

    /// Each reader WAS `std::env::var(NAME).ok().as_deref() == Some("1")` (the
    /// f32 seam's after its installed override) and now reads through its
    /// declaration, so the declaration must BE that historical predicate on
    /// every input: a wider or narrower kind would silently change a live gate.
    #[test]
    fn declarations_reproduce_the_exact_one_readers() {
        for &decl in FORWARD_LINEAR_LEVERS.decls() {
            for (raw, armed, source) in [
                (None, false, Source::Default),
                (Some("0"), false, Source::LegacyEnv),
                (Some("1"), true, Source::LegacyEnv),
                (Some("true"), false, Source::LegacyEnvRejected),
                (Some(" 1"), false, Source::LegacyEnvRejected),
                (Some(""), false, Source::LegacyEnvRejected),
            ] {
                let resolved = read_with(decl, |_| raw.map(str::to_owned));
                let legacy_reader = raw == Some("1");
                assert_eq!(resolved.value.as_bool(), armed, "{} {raw:?}", decl.name);
                assert_eq!(
                    resolved.value.as_bool(),
                    legacy_reader,
                    "{} {raw:?}: declaration disagrees with the raw reader",
                    decl.name
                );
                assert_eq!(resolved.source, source, "{} {raw:?}", decl.name);
            }
        }
    }

    #[test]
    fn the_module_declares_exactly_the_four_forward_linear_names() {
        let names: Vec<&str> = FORWARD_LINEAR_LEVERS
            .decls()
            .iter()
            .map(|decl| decl.name)
            .collect();
        assert_eq!(
            names,
            [
                NO_FORWARD_LINEAR_REF.name,
                NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF.name,
                FORWARD_LINEAR_SEQ_CONV_REF.name,
                FORWARD_LINEAR_F32.name,
            ]
        );
        for decl in FORWARD_LINEAR_LEVERS.decls() {
            assert!(
                crate::all().get(decl.name).is_some(),
                "{} must be filed in the merged registry",
                decl.name
            );
            assert_eq!(decl.bucket, Bucket::Debug, "{}", decl.name);
            assert!(
                matches!(decl.default, DefaultSpec::Bool(false)),
                "{}",
                decl.name
            );
        }
    }
}
