// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Collection and intermediate-tightening scheduling overrides.
//!
//! These declarations centralize the exact legacy parsers and make the
//! values visible in the flight receipt. Their current runtime consumers
//! still resolve process state directly (one is process-latched), so this is
//! migration preparation rather than a completed per-run `LeverSet` wiring.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const JOINT_ALPHA_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-joint-interm-alpha",
};

const WIDE_DEMANDED_INTERM_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-wide-demanded-interm-crown",
};

const COMPREHENSIVE_GPU_INTERM_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-comprehensive-gpu-interm-crown",
};

const PHASE_RESIDENT_CROWN_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-phase-resident-crown",
};

const CPU_PARALLEL_INTERM_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "root-cpu-parallel-interm-crown",
};

const WALK_ADMISSION_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "graph-crown-ibp-budget",
};

const COMPLETE_CLIP_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "complete-clip-host-reconstruct",
};

const ROOT_JOINT_MAX_DIM_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "override the legacy selector's 2,048-element contextual scope",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
               interm_refine.rs:scoped_joint_alpha_targets",
    },
    ReaderSite {
        scope: JOINT_ALPHA_SCOPE,
        role: "override the demand-ranked sound-GPU selector's 32,768-element \
               contextual scope",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
               interm_refine.rs:scoped_joint_alpha_targets_demand_ranked",
    },
];

declare_levers! {
    registry COLLECTION_LEVERS;

    /// `NY_ROOT_PHASE_RESIDENT_CROWN` — exact-one opt-in to the deferred,
    /// unified dense-head plus comprehensive resident root transaction.
    pub ROOT_PHASE_RESIDENT_CROWN = LeverDecl {
        name: "NY_ROOT_PHASE_RESIDENT_CROWN",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms one phase-resident sound-GPU root intermediate transaction whose ownership \
is resolved at the comprehensive slot and whose execution is deferred to the \
established dense-head slot after intervening prerequisites. The frozen request \
contains every structurally selected dense-head pre-activation and every row of \
those targets, plus the complete bounded comprehensive target census. Clean \
predispatch capacity declines may lower only comprehensive row ceilings through \
32, 16, and 8; dense rows and both target censuses remain fixed. The retained \
backend's exact typed preflight is final under an eight-GiB caller cap. Role-bound \
target identity, exact receipt validation, staged rowwise shrink-only publication, \
and one allocation-free commit make accepted execution atomic. Admission or a \
clean predispatch decline may use the established dense-head fallback once; no \
accepted failure may retry or fall back. Exact `1` opts in. Missing, exact `0`, \
malformed UTF-8, and non-UTF-8 values retain the shipped routes.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark exact-one Debug treatment; it reuses the retained sound-GPU \
                     request/receipt authority and has no claimed scored conversion",
        },
        owner: PHASE_RESIDENT_CROWN_SCOPE,
        readers: &[ReaderSite {
            scope: PHASE_RESIDENT_CROWN_SCOPE,
            role: "resolve exclusive ownership before deferring one unified root transaction",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
                   root.rs:root_phase_resident_crown_policy",
        }],
    };

    /// `NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN` — exact-one opt-in to the
    /// atomic all-target sound-WGPU root sweep.
    pub ROOT_COMPREHENSIVE_GPU_INTERM_CROWN = LeverDecl {
        name: "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms one comprehensive sound-GPU root intermediate sweep over every eligible \
demanded ReLU pre-activation in the bounded 2,048..=32,768 dimension class. \
Exact `1` opts in; absent, exact `0`, malformed UTF-8, and non-UTF-8 values \
retain the shipped OFF behavior. More than sixteen eligible targets, any \
structurally unpreparable target, or an unavailable automatic device resource \
profile refuses the whole route. Every predispatch retry retains every target \
and only lowers a common per-target row ceiling; both crossing and sign-stable \
non-point rows are retained when present. One frozen graph/bound transcript, \
one typed backend request, exact all-target result validation, and one staged \
shrink-only commit make every accepted result atomic. The backend profile is \
derived from live device class, granted buffer limits, and its existing memory \
budget rather than adapter names. This comprehensive gate owns the phase when \
armed, so refusal never falls through to the legacy one-target route. It has a \
twenty-second local authority slice and remains an unmeasured diagnostic: no \
speedup or scored-row conversion is claimed.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in; complete typed requests and \
                     publication are fail-closed and atomic, but no retained \
                     CIFAR value/verdict/deadline A/B exists yet",
        },
        owner: COMPREHENSIVE_GPU_INTERM_SCOPE,
        readers: &[ReaderSite {
            scope: COMPREHENSIVE_GPU_INTERM_SCOPE,
            role: "default-dark admission gate and owner of the comprehensive GPU root phase",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
                   root.rs:root_comprehensive_gpu_interm_crown_policy",
        }],
    };

    /// `NY_ROOT_CPU_PARALLEL_INTERM_CROWN` — exact-one opt-in to the atomic
    /// comprehensive host-CPU intermediate sweep.
    pub ROOT_CPU_PARALLEL_INTERM_CROWN = LeverDecl {
        name: "NY_ROOT_CPU_PARALLEL_INTERM_CROWN",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms one comprehensive root intermediate-bound CROWN transaction over every \
eligible ReLU pre-activation (dimension at most 16,384; at most ten targets). \
Exact `1` opts in; absent, exact `0`, malformed UTF-8, and non-UTF-8 values \
retain the shipped OFF behavior. More than ten eligible targets refuses the \
whole transaction rather than selecting a prefix. Independent target backwards \
share one immutable frozen map and one absolute deadline on a private host-only \
Rayon pool capped at four workers. Checked aggregate memory admission precedes \
snapshot allocation. CUDA/WGPU process-global f64 admission is structurally \
bypassed inside the pool; existing deadline-aware faer CPU folds remain \
authoritative. Every result is collected in canonical graph order and all \
L2-preserving shrink-only intersections are staged before one all-or-none map \
commit. Any refusal, expiry, disjoint interval, stale live map, or malformed \
result is a byte-identical no-op. The measured serial all-target diagnostic \
tightened 9/10 nodes in 349.1 s, reduced aggregate width 24,696.35 to 7,258.45, \
and moved the root census 0/99 to 3/99; the new parallel treatment itself has \
not yet completed a retained scored A/B, so it remains dark. Its local two-minute \
deadline is intentionally shorter than that 349.1 s serial measurement: this is \
a bounded diagnostic bridge, not a claim that CPU outer parallelism completes the \
all-target scored treatment or supplies the required speedup.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in; the complete transaction is \
                     fail-closed and shrink-only, but the CPU-parallel treatment \
                     has no retained scored all-target A/B yet",
        },
        owner: CPU_PARALLEL_INTERM_SCOPE,
        readers: &[ReaderSite {
            scope: CPU_PARALLEL_INTERM_SCOPE,
            role: "default-dark admission gate for the comprehensive atomic CPU sweep",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
                   comprehensive_cpu.rs:comprehensive_cpu_enabled",
        }],
    };

    /// `NY_ROOT_WIDE_DEMANDED_INTERM_CROWN` — exact-one opt-in to one bounded
    /// demanded wide intermediate sound fold.
    pub ROOT_WIDE_DEMANDED_INTERM_CROWN = LeverDecl {
        name: "NY_ROOT_WIDE_DEMANDED_INTERM_CROWN",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms one root intermediate-bound base CROWN fold for the highest-impact wide \
demanded ReLU pre-activation. Exact `1` opts in; absent, exact `0`, malformed \
UTF-8, and non-UTF-8 values retain the shipped OFF behavior. The reader fixes \
the first-slice limits at one target, dimensions 2,048 through 32,768, at most \
512 crossing rows, and eight seconds of dispatch/publication authority. Ranking \
and extraction poll that authority and a late preparation cannot dispatch or \
publish, but one synchronous extractor work unit is not preemptible. Ranking is selection-only: \
crossing unstable mass multiplied by the number of reachable unstable ReLU \
layers. Publication remains the existing certified zero-iteration sound-GPU \
fold followed by shrink-only intersection. The local backend must expose sound \
GpuCrownBackward and finite-deadline authority; every refusal is a no-op. This \
can move verdict-authoritative root boxes and spend scored wall time, so it ships \
dark until retained A/B evidence justifies promotion.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in; the shipped OFF arm is a \
                     bound/verdict-identical no-op, while the armed arm reuses certified \
                     shrink-only publication but has no retained track A/B yet",
        },
        owner: WIDE_DEMANDED_INTERM_SCOPE,
        readers: &[ReaderSite {
            scope: WIDE_DEMANDED_INTERM_SCOPE,
            role: "default-dark admission gate for the bounded one-target wide fold",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
                   root.rs:root_wide_demanded_interm_crown_policy",
        }],
    };

    /// `NY_CLIP_HOST_MEAN_LA` — exact-one opt-in to host mean-lA reconstruction.
    pub CLIP_HOST_MEAN_LA = LeverDecl {
        name: "NY_CLIP_HOST_MEAN_LA",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects the host mean-lA reconstruction used to rank Complete Clipping work \
when no admissible sound-GPU CROWN backend exists. Exact `1` opts in; absent, \
exact `0`, malformed UTF-8, and non-UTF-8 values retain the shipped OFF \
behavior. A contemporaneous relusplitter narrative reported that restored \
supply did not improve the 220-row result and cost branch throughput, but its \
machine-readable A/B receipts were not committed. That observation is useful \
diagnosis, not admissible Measured provenance. The snapshot is selection-only \
and published tightening remains independently certified, but the extra host \
CROWN passes consume the same absolute deadline and can therefore move bounds \
or verdicts; keep the default OFF and opt in only to investigate the downstream \
consumer when the current run has no admitted sound-GPU backend.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug override; the shipped OFF arm avoids \
                     all host reconstruction work, and no retained current-path \
                     A/B qualifies promotion",
        },
        owner: COMPLETE_CLIP_SCOPE,
        readers: &[ReaderSite {
            scope: COMPLETE_CLIP_SCOPE,
            role: "default-dark admission gate for host mean-lA reconstruction",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
                   interm_refine.rs:complete_clip_host_mean_la_enabled",
        }],
    };

    /// `NY_CLIP_INTERM_CERTIFIED` — exact-one opt-in to the certified batched clip.
    pub CLIP_INTERM_CERTIFIED = LeverDecl {
        name: "NY_CLIP_INTERM_CERTIFIED",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the certified batched intermediate-bound clip (interm_refine umbrella: \
`clip_resnet` + runtime `clip_guard`). Exact `1` opts in; absent, exact `0`, \
and every malformed spelling stay OFF. The armed path was reviewed 2026-08-13 \
(docs/CLIP_APPLY_ENCLOSURE_PROOF_DESIGN_2026-08-12.md): applied rows are \
minted CertifiedLayerCapture tokens, scope-validated per source and target, \
split half-spaces use the sound necessary-condition side in both directions, \
the coordinate solve carries an independent outward checker, and the unminted \
legacy lane is unreachable (zero production callers). Tightened caches feed \
verdict-authoritative CROWN bounds, so the blast radius is High; ships dark \
until the armed 220-row moat sweep lands its receipts in-tree. The legacy \
`NY_CLIP_INTERM`/`NY_CLIP_INTERM_RESNET` spellings remain permanently inert.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in; the shipped OFF arm is \
                     byte-identical to the historical quarantine, and the armed \
                     arm is gated on the reviewed certified bank path with the \
                     runtime guard retained as defense in depth",
        },
        owner: COMPLETE_CLIP_SCOPE,
        readers: &[ReaderSite {
            scope: COMPLETE_CLIP_SCOPE,
            role: "authority gate for the certified batched intermediate clip umbrella",
            site: "crates/ny-propagate/src/beta_crown/engine/graph/propagation/batched/\
                   interm_refine.rs:clip_interm_umbrella_enabled",
        }],
    };

    /// `NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM` — explicit selector-width override.
    pub ROOT_JOINT_INTERM_ALPHA_MAX_DIM = LeverDecl {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM",
        kind: LeverKind::UsizeTrimmed,
        // Absence means "use the reader's policy default": 2,048 for the
        // legacy selector and 32,768 for the demand-ranked GPU selector. A
        // single numeric declaration default would falsify one of those
        // contexts, so the receipt records this as an absent override.
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Overrides the maximum pre-activation dimension eligible for root joint \
intermediate-alpha selection. The legacy selector's contextual default is \
2,048; the demand-ranked sound-GPU selector's contextual default is 32,768. \
Surrounding whitespace accepted by Rust's `str::trim` is removed before \
parsing as `usize`, exactly \
matching both legacy readers. Malformed, negative, non-UTF-8, or \
platform-overflowing input is rejected and leaves the relevant contextual \
default in force. This changes only which already-sound tightening targets \
are attempted, but selection can move published bounds and deadline verdicts, \
so the blast radius is High. `DefaultSpec::Unset` deliberately describes an \
override rather than pretending the two consumers share one fallback.",
        provenance: Provenance::Unmeasured {
            why_ok: "explicit Debug override; selection is sound but no current \
                     authoritative A/B qualifies a shipped non-contextual value",
        },
        owner: JOINT_ALPHA_SCOPE,
        readers: ROOT_JOINT_MAX_DIM_READERS,
    };

    /// `NY_NO_WALK_RECORD_ADMISSION` — exact-one opt-out of record consultation.
    pub NO_WALK_RECORD_ADMISSION = LeverDecl {
        name: "NY_NO_WALK_RECORD_ADMISSION",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Exact `1` disables consultation of passive node-walk timing records by the \
Graph CROWN-IBP admission policy. Absent and exact `0` leave the default \
record-aware admission engaged; every other UTF-8 spelling is rejected and \
therefore also leaves that default engaged. Recording remains passive and \
always on. LEGACY-ARMED-UNQUALIFIED: the underlying record-aware policy ships \
engaged but has no discriminating current-path A/B; Phase 0 has no \
`DefaultStatus` field yet, so this exact marker keeps that debt explicit until \
the target schema can encode `LegacyArmedUnqualified`. Bucket::Debug classifies \
the explicit opt-out, not the evidence status of the underlying policy. The \
opt-out is useful for differential diagnosis, but changing a \
deadline admission decision can move published bounds or verdicts, so this \
remains a High-risk Debug lever.",
        provenance: Provenance::Unmeasured {
            why_ok: "explicit diagnostic opt-out around a tracked \
                     LEGACY-ARMED-UNQUALIFIED policy; both scheduling faces \
                     retain certified fallback behavior, but verdict/timing \
                     parity has not been established on the current sound path",
        },
        owner: WALK_ADMISSION_SCOPE,
        readers: &[ReaderSite {
            scope: WALK_ADMISSION_SCOPE,
            role: "process-latched opt-out of consulting completed/aborted walk records",
            site: "crates/ny-propagate/src/network/graph_alpha/bounds/\
                   budget_policy.rs:walk_record_admission_enabled",
        }],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_with, LeverValue, Source};

    #[test]
    fn max_dim_parser_preserves_trimmed_usize_contract() {
        for (raw, expected) in [("2048", 2_048_u64), (" 32768\t", 32_768_u64), ("+7", 7_u64)] {
            let resolved = read_with(&ROOT_JOINT_INTERM_ALPHA_MAX_DIM, |_| Some(raw.to_owned()));
            assert_eq!(resolved.value, LeverValue::U64(expected), "{raw:?}");
            assert_eq!(resolved.source, Source::LegacyEnv, "{raw:?}");
        }

        for raw in ["", "-1", "1.5", "all"] {
            let resolved = read_with(&ROOT_JOINT_INTERM_ALPHA_MAX_DIM, |_| Some(raw.to_owned()));
            assert_eq!(resolved.value, LeverValue::Unset, "{raw:?}");
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "{raw:?}");
        }
    }

    #[test]
    fn walk_record_opt_out_parser_is_exact() {
        for (raw, disabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&NO_WALK_RECORD_ADMISSION, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), disabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn host_mean_la_opt_in_preserves_default_off_exact_one_contract() {
        for (raw, enabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            // Malformed values fall back to the declared default (OFF), never
            // arm the lane by accident.
            (Some("true"), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&CLIP_HOST_MEAN_LA, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn wide_demanded_interm_opt_in_is_exact_and_default_off() {
        for (raw, enabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&ROOT_WIDE_DEMANDED_INTERM_CROWN, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn comprehensive_gpu_interm_opt_in_is_exact_and_default_off() {
        for (raw, enabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
            (Some(" 1"), false, Source::LegacyEnvRejected),
            (Some("1 "), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&ROOT_COMPREHENSIVE_GPU_INTERM_CROWN, |_| {
                raw.map(str::to_owned)
            });
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn phase_resident_crown_opt_in_is_exact_and_default_off() {
        for (raw, enabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
            (Some(" 1"), false, Source::LegacyEnvRejected),
            (Some("1 "), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&ROOT_PHASE_RESIDENT_CROWN, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[test]
    fn cpu_parallel_interm_opt_in_is_exact_and_default_off() {
        for (raw, enabled, source) in [
            (None, false, Source::Default),
            (Some("0"), false, Source::LegacyEnv),
            (Some("1"), true, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
            (Some(" 1"), false, Source::LegacyEnvRejected),
            (Some("1 "), false, Source::LegacyEnvRejected),
            (Some("01"), false, Source::LegacyEnvRejected),
            (Some("yes"), false, Source::LegacyEnvRejected),
            (Some(""), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&ROOT_CPU_PARALLEL_INTERM_CROWN, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value.as_bool(), enabled, "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn cpu_parallel_interm_non_utf8_value_is_present_but_rejected_off() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let registry = crate::collect(&[&COLLECTION_LEVERS]).expect("collection registry");
        let raw = crate::RawLeverInputs::capture_with(&registry, |name| {
            (name == ROOT_CPU_PARALLEL_INTERM_CROWN.name)
                .then(|| OsString::from_vec(vec![b'1', 0xff]))
        });
        let set = crate::LeverSet::resolve(&registry, &raw);
        let resolved = set
            .resolved(&ROOT_CPU_PARALLEL_INTERM_CROWN)
            .expect("CPU diagnostic lever registered");
        assert!(!resolved.value.as_bool());
        assert_eq!(resolved.source, Source::LegacyEnvRejected);
        assert_eq!(resolved.env_utf8, Some(false));
    }
}
