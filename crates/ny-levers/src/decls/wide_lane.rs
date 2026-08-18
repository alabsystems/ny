// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Wide-lane routing and authoritative-deadline overrides.
//!
//! These selectors can change which certified kernel produces a bound or how
//! much work is admitted before an authoritative deadline. They therefore
//! remain High-risk Debug levers even though every lane retains a sound
//! fallback.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const RESNET_WIDE_SCOPE: Scope = Scope {
    package: "ny-gpu",
    subsystem: "resnet-wide-crown",
};

const MO_CHUNK_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "multi-objective-gpu-chunk",
};

declare_levers! {
    registry WIDE_LANE_LEVERS;

    /// `NY_BAB_RESNET_WIDE` — legacy-armed wide ResNet CROWN kernel.
    pub BAB_RESNET_WIDE = LeverDecl {
        name: "NY_BAB_RESNET_WIDE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects the resident wide ResNet CROWN kernel for multi-domain batches. Exact \
`0` opts out; absence, exact `1`, malformed UTF-8, and non-UTF-8 values retain \
the shipped ON behavior. Every shape, build, dispatch, or enclosure refusal \
falls through to the proven per-domain stacker. LEGACY-ARMED-UNQUALIFIED: the \
shipped default remains on for behavior compatibility, but the live-GPU lane \
does not yet have a discriminating current-path A/B qualifying it as a \
DefaultOn policy. Bucket::Debug classifies the exact-zero diagnostic opt-out, \
not approval of the underlying default. The kernel can move verdict-bearing \
floating-point bounds, so this remains High-risk despite its sound fallback.",
        provenance: Provenance::Unmeasured {
            why_ok: "tracked LEGACY-ARMED-UNQUALIFIED kernel selector; device \
                     enclosure oracles establish soundness but do not establish \
                     deadline, value, or verdict parity for the shipped lane",
        },
        owner: RESNET_WIDE_SCOPE,
        readers: &[
            ReaderSite {
                scope: RESNET_WIDE_SCOPE,
                role: "select the wide single-pass kernel before the per-domain fallback",
                site: "crates/ny-gpu/src/wgpu_device/ops/crown_backward.rs:\
                       crown_backward_gpu_resnet_sound_beta_batched",
            },
            ReaderSite {
                scope: RESNET_WIDE_SCOPE,
                role: "make the global wide-kernel opt-out outrank heterogeneous subgrouping",
                site: "crates/ny-gpu/src/wgpu_device/ops/crown_backward.rs:\
                       try_wide_resnet_batched_subgrouped",
            },
        ],
    };

    /// `NY_BAB_RESNET_WIDE_SUBGROUP` — dark heterogeneous-wave subgrouping.
    pub BAB_RESNET_WIDE_SUBGROUP = LeverDecl {
        name: "NY_BAB_RESNET_WIDE_SUBGROUP",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact `1` lets a heterogeneous ResNet batch be partitioned into maximal \
homogeneous runs and folded wide run by run. Absence, exact `0`, malformed \
UTF-8, and non-UTF-8 values retain the historical whole-wave fallback. The \
global `NY_BAB_RESNET_WIDE=0` kill switch still outranks this selector. \
Preflight validates every run before dispatch so any refusal falls closed \
without publishing a partial wide result. The lane remains dark and \
High-risk because it changes which kernel produces verdict-bearing bounds.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one diagnostic lane with a fail-closed serial \
                     fallback; the device enclosure oracle is not a qualifying \
                     target-adapter value/verdict/deadline A/B",
        },
        owner: RESNET_WIDE_SCOPE,
        readers: &[ReaderSite {
            scope: RESNET_WIDE_SCOPE,
            role: "admit heterogeneous-wave partitioning before any wide dispatch",
            site: "crates/ny-gpu/src/wgpu_device/ops/crown_backward.rs:\
                   wide_subgroup_enabled",
        }],
    };

    /// `NY_MO_GPU_CHUNK_DEADLINE` — dark scored-deadline chunk override.
    pub MO_GPU_CHUNK_DEADLINE = LeverDecl {
        name: "NY_MO_GPU_CHUNK_DEADLINE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact `1` permits `NY_MO_GPU_CHUNK` to raise the wide multi-objective GPU batch \
width even when an authoritative deadline is present. Absence, exact `0`, \
malformed UTF-8, and non-UTF-8 values retain the measured eight-domain cap. \
The cooperative backend deadline remains a second line of defense, but a \
wider atomic chunk can overrun the inter-chunk check and change which \
certified bounds arrive before the verdict deadline. The override is therefore \
dark, High-risk, and unqualified for a shipped default.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one scheduling override; all returned bounds \
                     remain certified, but scored deadline and verdict parity \
                     have not been established",
        },
        owner: MO_CHUNK_SCOPE,
        readers: &[ReaderSite {
            scope: MO_CHUNK_SCOPE,
            role: "permit an explicit GPU chunk width above the scored-deadline cap",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/\
                   multi_objective/batched/batched_multi.rs:\
                   deadline_chunk_override_enabled",
        }],
    };

    /// `NY_KFSB_SIM_SHARE` — the advisory kFSB simulation's share of the BaB round.
    pub KFSB_SIM_SHARE = LeverDecl {
        name: "NY_KFSB_SIM_SHARE",
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: 1.0 },
        default: DefaultSpec::F64(0.35),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Fraction of the remaining BaB round the kFSB SIMULATION may spend RANKING \
candidate splits before it stops. The simulation previously had no budget of its \
own — its chunk loop broke only on the global BaB deadline — so ranking could \
consume the round while the children it ranked went unevaluated. Whitespace is \
trimmed; the closed interval [0, 1] is admissible and `0` is the documented kill \
switch that restores the previous unbounded behaviour. Malformed, out-of-range \
and non-UTF-8 values retain the 0.35 default.

Stopping ranking early means choosing from FEWER candidates, so a wave may \
commit a worse split — a scheduling and quality effect, not a soundness one. It \
cannot produce a wrong bound: every committed child is still evaluated by the \
ordinary sound backward, and an unranked candidate is simply not chosen. Hence \
MoatRisk::Low rather than High.",
        provenance: Provenance::Measured {
            commit: "d6e27c392",
            date: "2026-08-16",
            artifact: "commit d6e27c392 message (cifar100 idx_8600, official 100 s budget)",
            delta: "BaB round 1 wall clock 9.35 s -> 3.95 s (2.4x) with sims 384 -> 192, \
                    on top of the previous commit's 6.2x per-candidate forward. Moat 12/12 \
                    banked cifar100 verdicts preserved. ZERO conversions: the decisive \
                    counter `explored=1 verified=0 queue=16 max_depth=0` is unchanged \
                    across idx_8600 / idx_2176 / idx_2132, because freeing 5.4 s did not \
                    buy a second BaB expansion.",
        },
        owner: MO_CHUNK_SCOPE,
        readers: &[ReaderSite {
            scope: MO_CHUNK_SCOPE,
            role: "bound the advisory split-ranking simulation to a share of the round",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/\
                   multi_objective/batched/kfsb_multi.rs (sim_deadline)",
        }],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_with, Source};

    #[test]
    fn wide_kernel_parser_preserves_exact_zero_legacy_kill_switch() {
        for (raw, enabled, source) in [
            (None, true, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), true, Source::LegacyEnvRejected),
            (Some(""), true, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&BAB_RESNET_WIDE, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn dark_wide_selectors_preserve_exact_one_contract() {
        for decl in [&BAB_RESNET_WIDE_SUBGROUP, &MO_GPU_CHUNK_DEADLINE] {
            for (raw, enabled, source) in [
                (None, false, Source::Default),
                (Some("0"), false, Source::LegacyEnv),
                (Some("1"), true, Source::LegacyEnv),
                (Some("true"), false, Source::LegacyEnvRejected),
                (Some(""), false, Source::LegacyEnvRejected),
            ] {
                let resolved = read_with(decl, |_| raw.map(str::to_owned));
                assert_eq!(resolved.value.as_bool(), enabled, "{} {raw:?}", decl.name);
                assert_eq!(resolved.source, source, "{} {raw:?}", decl.name);
            }
        }
    }

    /// Pins the CLOSED interval and the trimming, which is why this axis is not
    /// `F64Open`. `0` is the documented kill switch that restores unbounded
    /// ranking; under an open interval it would be rejected and silently resolve
    /// to 0.35, arming the very cap it is meant to remove.
    #[test]
    fn kfsb_sim_share_admits_both_endpoints_and_trims() {
        for (raw, value, source) in [
            (None, 0.35, Source::Default),
            (Some("0"), 0.0, Source::LegacyEnv),
            (Some("1"), 1.0, Source::LegacyEnv),
            (Some("0.5"), 0.5, Source::LegacyEnv),
            (Some(" 0.5 "), 0.5, Source::LegacyEnv),
            // Out of range and malformed both fall back to the shipped fraction.
            (Some("-0.1"), 0.35, Source::LegacyEnvRejected),
            (Some("1.1"), 0.35, Source::LegacyEnvRejected),
            (Some("nan"), 0.35, Source::LegacyEnvRejected),
            (Some("inf"), 0.35, Source::LegacyEnvRejected),
            (Some("half"), 0.35, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&KFSB_SIM_SHARE, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_f64(), Some(value), "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }
}
